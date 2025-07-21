# wavelet.py
# Wavelet Animation Script
# This script generates an animated visualization of wavelet decomposition
# using various wavelet types and function types. It allows customization
# of parameters such as wavelet type, function type, number of wavelets,
# and animation speed. The animation can be saved as a GIF or MP4 file.	
#


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pywt
import argparse
import sys
import os
import datetime

# Configuration parameters
parser = argparse.ArgumentParser(description="Wavelet animation parameters")
parser.add_argument('--frames_per_wavelet', type=int, default=12, help='Animation speed (frames per wavelet)')
parser.add_argument('--wavelet_type', type=str, default='haar', 
				   help='Wavelet type. Options: db4, db6, db8, haar, bior2.2, bior4.4, coif2, coif4, sym4, sym8, dmey. Default: haar')
parser.add_argument('--function_type', type=str, default='smooth_periodic', 
				   help='Function type. Options: smooth, piecewise_linear, discontinuous, smooth_periodic, mix. Default: smooth_periodic')
parser.add_argument('--function_seed', type=int, default=38324, help='Random seed for function generation. Default: 38324')
parser.add_argument('--number_wavelets', type=int, default=64, help='Number of wavelets to animate. Default: 64')
parser.add_argument('--save', type=str, default='none', help='Save animation as either GIF or MP4. Options: gif, mp4, none. Default: none')

args = parser.parse_args()

frames_per_wavelet = args.frames_per_wavelet
wavelet_type = args.wavelet_type
function_type = args.function_type
function_seed = args.function_seed 
number_max_wavelets_animation = args.number_wavelets
save = args.save


def record_command():
		"""Record the command used to run this script"""
		try:
			# Get the command line arguments
			command_parts = ['python wavelet.py'] + sys.argv[1:]
			command_str = ' '.join(command_parts)
			
			# Create a log file in the same directory as the script
			script_dir = os.path.dirname(os.path.abspath(__file__))
			log_file = os.path.join(script_dir, 'wavelet_history.log')
			
			# Append the command to the log file with timestamp
			timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			
			with open(log_file, 'a') as f:
				f.write(f"[{timestamp}] {command_str}\n")
			
			# Keep only last 100 commands
			with open(log_file, 'r') as f:
				lines = f.readlines()
			
			if len(lines) > 100:
				with open(log_file, 'w') as f:
					f.writelines(lines[-100:])
					
		except Exception as e:
			print(f"Warning: Could not record command - {e}")

record_command()

# [Previous functions remain the same until animation setup]

# Gera uma função aleatória suave em [0,1]
def random_function(n_points=512, function_type='smooth'):
	"""
	Generate different types of functions for wavelet analysis
	
	Args:
		n_points: Number of points in the function
		function_type: Type of function - 'smooth','piecewise_linear', 'smooth_periodic', 'discontinuous', 'mix'
	"""
	np.random.seed(function_seed)  # Set random seed for reproducibility
	x = np.linspace(0, 2 * np.pi, n_points)
	
	if function_type == 'smooth':
		# Smooth function using sum of sines
		result = np.zeros(n_points)
		for i in range(1, 10):
			amplitude = 1.0 / (i ** 1.5)
			frequency = i + np.random.uniform(-0.5, 0.5)
			phase = np.random.uniform(0, 2 * np.pi)
			result += amplitude * np.sin(frequency * x + phase)
	
	elif function_type == 'piecewise_linear':
		# Piecewise linear function
		n_segments = 8
		breakpoints = np.sort(np.random.uniform(0, 2*np.pi, n_segments-1))
		breakpoints = np.concatenate([[0], breakpoints, [2*np.pi]])
		values = np.random.uniform(-1, 1, len(breakpoints))
		result = np.interp(x, breakpoints, values)
	
	elif function_type == 'discontinuous':
		# Function with discontinuous jumps
		result = np.zeros(n_points)
		n_jumps = 6
		jump_positions = np.sort(np.random.choice(n_points, n_jumps, replace=False))
		
		current_level = 0
		for i, pos in enumerate(jump_positions):
			if i == 0:
				result[:pos] = current_level
			else:
				result[jump_positions[i-1]:pos] = current_level
			current_level += np.random.uniform(-0.5, 0.5)
		result[jump_positions[-1]:] = current_level
		
		# Add some smooth variation within segments
		smooth_component = 0.2 * np.sin(3 * x + np.random.uniform(0, 2*np.pi))
		result += smooth_component
	elif function_type == 'smooth_periodic':
		# Smooth periodic function
		x = np.linspace(0, 2 * np.pi, n_points)
	
		# Soma de várias componentes senoidais com amplitudes decrescentes
		result = np.zeros(n_points)
		for i in range(1, 10):  # 7 componentes harmônicas
			amplitude = 1.0 / (i ** -1.5)  # Amplitudes decrescentes
			frequency = i + np.random.uniform(-4, 4)  # Frequência com ruído
			phase = np.random.uniform(0, 2 * np.pi)  # Fase aleatória
			result += amplitude * np.sin(frequency * x + phase)
	
		# Normaliza para [0,1]
		result = (result - result.min()) / (result.max() - result.min())
	elif function_type == 'mix':
		# Mix of smooth, piecewise, and discontinuous features
		# Smooth base
		smooth_part = np.sum([0.3 * np.sin((i+1) * x + np.random.uniform(0, 2*np.pi)) / (i+1) 
							 for i in range(5)], axis=0)
		
		# Add some discontinuous jumps
		jump_pos = n_points // 3
		result = smooth_part.copy()
		result[jump_pos:] += 0.4
		
		# Add piecewise linear component
		linear_x = np.linspace(0, 1, 4)
		linear_y = np.random.uniform(-0.2, 0.2, 4)
		linear_component = 0.3 * np.interp(x / (2*np.pi), linear_x, linear_y)
		result += linear_component
	
	else:
		raise ValueError("function_type must be 'smooth','smooth_periodic',  'piecewise_linear', 'discontinuous', or 'mix'")

	# Normalize to [0,1]
	result = (result - result.min()) / (result.max() - result.min())
	return result

# Prepara os dados
n_points = 512
x = np.linspace(0, 1, n_points)
f = random_function(n_points, function_type)

# Decomposição usando wavelet especificada
coeffs = pywt.wavedec(f, wavelet_type, mode='periodization')

# Calculate total wavelets and limit to max
total_wavelets_available = sum(len(coeffs[i]) for i in range(0, len(coeffs)))
max_wavelets = min(number_max_wavelets_animation, total_wavelets_available)

print(f"Total wavelets available: {total_wavelets_available}")
print(f"Animating first {max_wavelets} wavelets")

# Configuração da figura - reduced DPI for memory efficiency
fig, ax = plt.subplots(figsize=(5, 8), dpi=100)
line_original, = ax.plot(x, f, 'gray', linewidth=2, alpha=0.7, label='Original')
line_recon, = ax.plot(x, np.zeros_like(x), 'b', label='Reconstruction')
ax.legend(loc='lower left')
ax.set_ylim(-0.5, 1)
ax.set_aspect('equal')

# Animação
def update(frame):
	# Cria uma cópia dos coeficientes
	coeffs_partial = [c.copy() for c in coeffs]
	single_wavelet_coeffs = [np.zeros_like(c) for c in coeffs]
	
	# Zera todos os níveis de detalhe
	for i in range(0, len(coeffs_partial)):
		coeffs_partial[i] = np.zeros_like(coeffs_partial[i])
	
	# Calcula quantos coeficientes wavelet já foram adicionados (limited to max_wavelets)
	wavelets_to_add = min(frame // frames_per_wavelet, max_wavelets)
	
	# Adiciona wavelets uma por vez
	wavelet_count = 0
	for level in range(0, len(coeffs)):
		level_size = len(coeffs[level])
		
		if wavelet_count + level_size <= wavelets_to_add:
			# Adiciona o nível completo
			coeffs_partial[level] = coeffs[level].copy()
			wavelet_count += level_size
			if wavelet_count == wavelets_to_add:
				t = (frame % frames_per_wavelet) / (frames_per_wavelet - 1.0)
				coeffs_partial[level][level_size-1] = coeffs_partial[level][level_size-1]*t
				single_wavelet_coeffs[level][level_size-1] = coeffs[level][level_size-1]
				break
		elif wavelet_count < wavelets_to_add:
			# Adiciona parcialmente este nível
			remaining = wavelets_to_add - wavelet_count
			t = (frame % frames_per_wavelet) / (frames_per_wavelet - 1.0)
			coeffs_partial[level][:remaining-1] = coeffs[level][:remaining-1]
			coeffs_partial[level][remaining-1] = coeffs[level][remaining-1] * t
			# Draw the wavelet being added in orange
			# Reconstruct just this single wavelet
			single_wavelet_coeffs[level][remaining-1] = coeffs[level][remaining-1]
			break
		else:
			break
	
	f_reconstructed = pywt.waverec(coeffs_partial, wavelet_type, mode='periodization')
	
	line_recon.set_ydata(f_reconstructed)

	single_wavelet = pywt.waverec(single_wavelet_coeffs, wavelet_type, mode='periodization')
			
	
	# Clear previous fill and add new transparent marine blue fill
	for coll in ax.collections[:]:
		coll.remove()
	ax.fill_between(x, 0, f_reconstructed, alpha=0.3, color='steelblue')
	# Clear previous orange lines
	for line in ax.lines[2:]:  # Keep original and reconstruction lines
		line.remove()
	if np.any(single_wavelet != 0):
		ax.fill_between(x, 0, single_wavelet, alpha=0.7, color='orange', edgecolor='darkorange')
	
	progress_pct = (wavelets_to_add / max_wavelets) * 100
	ax.set_title(f'Adding wavelets ({wavelet_type}): {wavelets_to_add}/{max_wavelets} ({progress_pct:.1f}%)')
	
	return line_recon,

# Total de frames (limited to max_wavelets)
total_frames = max_wavelets * frames_per_wavelet
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

# Save the animation as GIF with optimized settings
if save:
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	parameters = f"{wavelet_type}_{function_type}_{max_wavelets}w_{function_seed}s_{frames_per_wavelet}f"
	
	if save.lower() == 'gif':
		filename = f'wavelet-{timestamp}-{parameters}.gif'
		print(f"Saving GIF: {filename}")
		print("This may take a while for memory optimization...")
		
		# Use optimized settings for smaller file size and less memory usage
		ani.save(filename, writer='pillow', fps=15, 
				savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
		print(f"GIF saved as {filename}")
	
	elif save.lower() == 'mp4':
		filename = f'wavelet-{timestamp}-{parameters}.mp4'
		print(f"Saving MP4: {filename}")
		print("This may take a while...")
		
		# Use ffmpeg writer for MP4
		ani.save(filename, writer='ffmpeg', fps=20, bitrate=1800,
				savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
		print(f"MP4 saved as {filename}")
	
	else:
		print(f"Unsupported format: {save}. Use 'gif' or 'mp4'")

plt.show()
