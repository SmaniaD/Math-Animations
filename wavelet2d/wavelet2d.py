# wavelet2d.py
# 2D Wavelet Animation Script
# This script generates an animated visualization of 2D wavelet decomposition
# using various wavelet types and function types. It shows both the original
# 2D function and its progressive reconstruction using wavelets. The animation
# can be displayed in landscape or portrait orientation and saved as GIF or MP4.
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pywt
import argparse
import subprocess
import sys
import os
import datetime
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Configuration parameters
parser = argparse.ArgumentParser(description="2D Wavelet animation parameters")
parser.add_argument('--frames_per_wavelet', type=int, default=3, help='Animation speed (frames per wavelet)')
parser.add_argument('--wavelet_type', type=str, default='haar', 
				   help='Wavelet type. Options: ' + ', '.join(pywt.wavelist(kind='discrete')) + '. Default: haar')
parser.add_argument('--function_type', type=str, default='smooth', 
				   help='Function type. Options: smooth, piecewise_linear, discontinuous, smooth_periodic, mix, dirac. Default: smooth')
parser.add_argument('--function_seed', type=int, default=38324, help='Random seed for function generation. Default: 38324')
parser.add_argument('--number_wavelets', type=int, default=256, help='Number of wavelets to animate. Default: 256')
parser.add_argument('--save', type=str, choices=['gif', 'mp4'], default='none', help='Save animation as GIF or MP4. Options: gif, mp4')
parser.add_argument('--grid_size', type=int, default=64, help='Grid size for 2D function. Default: 64')
parser.add_argument('--color_scheme', type=str, default='marine', 
				   help='Color scheme for reconstruction. Options: vibrant, gray, marine, ocean. Default: marine')
parser.add_argument('--original_style', type=str, default='wireframe', 
				   help='Original function style. Options: transparent, wireframe, solid_contrast, gradient, oscillating_transparent, none. Default: wireframe')
parser.add_argument('--original_alpha', type=float, default=0.9, help='Base transparency of original function (0.0-1.0). Default: 0.9')
parser.add_argument('--oscillating_speed', type=float, default=0.05, help='Oscillation speed for original function transparency. Default: 0.05')
parser.add_argument('--orientation', type=str, default='landscape', 
				   help='Layout orientation. Options: landscape (side-by-side), portrait (vertical stack). Default: landscape')
parser.add_argument('--camera', type=str, default='rotating', help='Moviment of camera. Options: static_overview, rotating. Default: rotating')
parser.add_argument('--single_wavelet',  type=str, default='3d', help='Animate a single 2d or 3d. Options: 2d, 3d. Default: 3d')

args = parser.parse_args()

frames_per_wavelet = args.frames_per_wavelet
wavelet_type = args.wavelet_type
function_type = args.function_type
function_seed = args.function_seed 
number_max_wavelets_animation = args.number_wavelets
save = args.save
grid_size = args.grid_size
color_scheme = args.color_scheme
original_style = args.original_style
original_alpha = args.original_alpha
original_oscillation_speed = args.oscillating_speed
orientation = args.orientation
camera = args.camera
single_wavelet_rendering = args.single_wavelet


def record_command():
		"""Record the command used to run this script"""
		try:
			# Get the command line arguments
			command_parts = ['python wavelet2d.py'] + sys.argv[1:]
			command_str = ' '.join(command_parts)
			
			# Create a log file in the same directory as the script
			script_dir = os.path.dirname(os.path.abspath(__file__))
			log_file = os.path.join(script_dir, 'wavelet2d_history.log')
			
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


def random_function_2d(n_points=64, function_type='smooth'):
	"""
	Generate different types of 2D functions for wavelet analysis
	
	Args:
		n_points: Number of points along each axis
		function_type: Type of function - 'smooth','piecewise_linear', 'smooth_periodic', 'discontinuous', 'mix'
	"""
	np.random.seed(function_seed)
	x = np.linspace(0, 2 * np.pi, n_points)
	y = np.linspace(0, 2 * np.pi, n_points)
	X, Y = np.meshgrid(x, y)

	if function_type == 'dirac':
		# Dirac delta function in 2D
		result = np.zeros((n_points, n_points))
		center = n_points // 2
		result[center, center] = 100000.0
	
	elif function_type == 'smooth':
		# Smooth 2D function using sum of 2D sines
		result = np.zeros((n_points, n_points))
		for i in range(1, 3):
			for j in range(1, 3):
				amplitude = 1.0 / ((i * j) ** 0.5)
				freq_x = i + np.random.uniform(-0.5, 0.5)
				freq_y = j + np.random.uniform(-0.5, 0.5)
				phase = np.random.uniform(0, 2 * np.pi)
				result += amplitude * np.sin(freq_x * X + freq_y * Y + phase)
	
	elif function_type == 'smooth_periodic':
		# Smooth periodic 2D function
		result = np.zeros((n_points, n_points))
		for i in range(1, 6):
			for j in range(1, 6):
				amplitude = 1.0 / ((i + j) ** 1.2)
				freq_x = i + np.random.uniform(-2, 2)
				freq_y = j + np.random.uniform(-2, 2)
				phase = np.random.uniform(0, 2 * np.pi)
				result += amplitude * np.sin(freq_x * X + freq_y * Y + phase)
	
	elif function_type == 'discontinuous':
		# 2D function with discontinuous regions
		result = np.zeros((n_points, n_points))
		n_regions = 4
		for _ in range(n_regions):
			center_x = np.random.randint(n_points//4, 3*n_points//4)
			center_y = np.random.randint(n_points//4, 3*n_points//4)
			radius = np.random.randint(n_points//8, n_points//4)
			value = np.random.uniform(-1, 1)
			
			for i in range(n_points):
				for j in range(n_points):
					if (i - center_x)**2 + (j - center_y)**2 < radius**2:
						result[i, j] = value
	
	elif function_type == 'piecewise_linear':
		# Piecewise linear 2D function
		result = np.random.uniform(-0.5, 0.5, (n_points, n_points))
		# Add some smooth interpolation
		from scipy.ndimage import gaussian_filter

		result = gaussian_filter(result, sigma=n_points//16)
	
	elif function_type == 'mix':
		# Mix of different features
		# Smooth base
		smooth_part = np.sin(2*X) * np.cos(Y) + 0.5*np.sin(X*Y)
		
		# Add discontinuous region
		center = n_points // 2
		radius = n_points // 6
		mask = (X - x[center])**2 + (Y - y[center])**2 < (x[radius] - x[0])**2
		result = smooth_part.copy()
		result[mask] += 0.8
	
	else:
		raise ValueError("function_type must be 'smooth','smooth_periodic', 'piecewise_linear', 'discontinuous', or 'mix'")

	# Normalize to [0,1]
	result = (result - result.min()) / (result.max() - result.min())
	return result

# Generate 2D function
f_2d = random_function_2d(grid_size, function_type)

# 2D Wavelet decomposition
coeffs_2d = pywt.wavedec2(f_2d, wavelet_type, mode='periodization')

# Count total wavelets
total_wavelets_available = 0
for i, level_coeffs in enumerate(coeffs_2d):
	if i == 0:  # First level is approximation coefficients only
		total_wavelets_available += level_coeffs.size
	else:  # Subsequent levels have detail coefficients (cH, cV, cD)
		cH, cV, cD = level_coeffs
		total_wavelets_available += cH.size + cV.size + cD.size

max_wavelets = min(number_max_wavelets_animation, total_wavelets_available)

print(f"Grid size: {grid_size}x{grid_size}")
print(f"Total wavelets available: {total_wavelets_available}")
print(f"Animating first {max_wavelets} wavelets")
print(f"Color scheme: {color_scheme}")
print(f"Original function style: {original_style}")
print(f"Original function alpha: {original_alpha}")
print(f"Orientation: {orientation}")

# Create custom colormaps based on color_scheme
def get_colormaps(scheme):
	if scheme == 'vibrant':
		colors_main = ['#000033', '#000066', '#0066CC', '#00CCFF', '#66FFCC', '#CCFF66', '#FFCC00', '#FF6600', '#FF0000']
		colors_wavelet = ['#330000', '#660000', '#CC0066', '#FF00CC', '#FF66FF', '#FFCCFF']
		colors_original = ['#4a0080', '#8000ff', '#bf80ff', '#dfbfff']
	elif scheme == 'gray':
		colors_main = ['#000000', '#1a1a1a', '#333333', '#4d4d4d', '#666666', '#808080', '#999999', '#b3b3b3', '#cccccc', '#ffffff']
		colors_wavelet = ['#2d0000', '#4d0000', '#660000', '#800000', '#990000', '#cc0000']
		colors_original = ['#ff4500', '#ff6347', '#ff7f50', '#ffa07a']
	elif scheme == 'marine':
		colors_main = ['#001122', '#003366', '#004080', '#0066cc', '#3399ff', '#66ccff', '#99ddff', '#ccf0ff']
		colors_wavelet = ['#ff6600', '#ff8833', '#ffaa66', '#ffcc99', '#ffe5cc']
		colors_original = ['#8b0000', '#dc143c', '#ff69b4', '#ffb6c1']
	elif scheme == 'ocean':
		colors_main = ['#000033', '#000080', '#0033cc', '#0066ff', '#3399ff', '#66ccff', '#99e6ff', '#ccf5ff']
		colors_wavelet = ['#ff4500', '#ff6347', '#ff7f50', '#ffa07a', '#ffb6c1']
		colors_original = ['#2e8b57', '#3cb371', '#90ee90', '#98fb98']
	else:
		raise ValueError("color_scheme must be 'vibrant', 'gray', 'marine', or 'ocean'")
	
	cmap_main = LinearSegmentedColormap.from_list(f'{scheme}_main', colors_main, N=256)
	cmap_wavelet = LinearSegmentedColormap.from_list(f'{scheme}_wavelet', colors_wavelet, N=256)
	cmap_original = LinearSegmentedColormap.from_list(f'{scheme}_original', colors_original, N=256)
	
	return cmap_main, cmap_wavelet, cmap_original

cmap_main, cmap_wavelet, cmap_original = get_colormaps(color_scheme)

# Setup visualization with orientation-dependent layout
plt.style.use('dark_background')

if orientation == 'portrait':
	# Portrait layout (smartphone-like proportions)
	fig = plt.figure(figsize=(9, 16), facecolor='black')
	ax1 = fig.add_subplot(211, projection='3d', facecolor='black')  # Reconstruction on top
	ax2 = fig.add_subplot(212, projection='3d', facecolor='black')  # Original on bottom
	layout_name = 'portrait'
else:
	# Landscape layout (side-by-side)
	fig = plt.figure(figsize=(20, 10), facecolor='black')
	ax1 = fig.add_subplot(121, projection='3d', facecolor='black')  # Original function
	ax2 = fig.add_subplot(122, projection='3d', facecolor='black')  # Reconstruction
	layout_name = 'landscape'

# Function to setup axis styling
def setup_axis(ax, title):
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False
	ax.xaxis.pane.set_edgecolor('white')
	ax.yaxis.pane.set_edgecolor('white')
	ax.zaxis.pane.set_edgecolor('white')
	ax.grid(True, color='gray', alpha=0.3)
	ax.set_xlabel('X', color='white', fontsize=12)
	ax.set_ylabel('Y', color='white', fontsize=12)
	ax.set_zlabel('Z', color='white', fontsize=12)
	ax.set_zlim(0, 1)
	ax.tick_params(colors='white')
	ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=20)

# Setup both axes with orientation-dependent titles
if orientation == 'portrait':
	setup_axis(ax1, 'Wavelet Reconstruction')
	setup_axis(ax2, 'Original Function')
else:
	setup_axis(ax1, 'Original Function')
	setup_axis(ax2, 'Wavelet Reconstruction')

# Create coordinate grids
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, y)

# Initial plots
if orientation == 'portrait':
	surface1 = ax1.plot_surface(X, Y, np.zeros_like(X), cmap=cmap_main, alpha=0.9)
	surface2 = ax2.plot_surface(X, Y, f_2d, cmap=cmap_original, alpha=0.9)
else:
	surface1 = ax1.plot_surface(X, Y, f_2d, cmap=cmap_original, alpha=0.9)
	surface2 = ax2.plot_surface(X, Y, np.zeros_like(X), cmap=cmap_main, alpha=0.9)

def update(frame):
	global surface1, surface2
	
	# Create partial coefficients
	coeffs_partial = []
	single_wavelet_coeffs = []
	
	for level_idx, level_coeffs in enumerate(coeffs_2d):
		if level_idx == 0:  # Approximation coefficients (first level)
			coeffs_partial.append(np.zeros_like(level_coeffs))
			single_wavelet_coeffs.append(np.zeros_like(level_coeffs))
		else:  # Detail coefficients (subsequent levels)
			cH, cV, cD = level_coeffs
			coeffs_partial.append((np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))
			single_wavelet_coeffs.append((np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))
	
	# Calculate how many wavelets to add
	wavelets_to_add = min(frame // frames_per_wavelet, max_wavelets)
	t = (frame % frames_per_wavelet) / max(frames_per_wavelet - 1.0, 1.0)
	
	# Add wavelets progressively
	current_wavelet_idx = 0
	
	for level_idx, level_coeffs in enumerate(coeffs_2d):
		if level_idx == 0:  # Approximation coefficients (first level)
			for i in range(level_coeffs.shape[0]):
				for j in range(level_coeffs.shape[1]):
					if current_wavelet_idx < wavelets_to_add:
						if current_wavelet_idx == wavelets_to_add - 1:
							coeffs_partial[level_idx][i, j] = level_coeffs[i, j] * t
							single_wavelet_coeffs[level_idx][i, j] = level_coeffs[i, j]
						else:
							coeffs_partial[level_idx][i, j] = level_coeffs[i, j]
						current_wavelet_idx += 1
					else:
						break
				if current_wavelet_idx >= wavelets_to_add:
					break
		else:  # Detail coefficients (subsequent levels)
			cH, cV, cD = level_coeffs
			
			# Handle detail coefficients
			for detail_idx, detail in enumerate([cH, cV, cD]):
				if current_wavelet_idx >= wavelets_to_add:
					break
				for i in range(detail.shape[0]):
					for j in range(detail.shape[1]):
						if current_wavelet_idx < wavelets_to_add:
							if current_wavelet_idx == wavelets_to_add - 1:
								coeffs_partial[level_idx][detail_idx][i, j] = detail[i, j] * t
								single_wavelet_coeffs[level_idx][detail_idx][i, j] = detail[i, j]
							else:
								coeffs_partial[level_idx][detail_idx][i, j] = detail[i, j]
							current_wavelet_idx += 1
						else:
							break
					if current_wavelet_idx >= wavelets_to_add:
						break
	
	# Reconstruct the function
	f_reconstructed = pywt.waverec2(coeffs_partial, wavelet_type, mode='periodization')
	
	# Clear previous surfaces
	ax1.clear()
	ax2.clear()
	
	# Reset styling after clear with orientation-dependent titles
	if orientation == 'portrait':
		setup_axis(ax1, 'Wavelet Reconstruction')
		setup_axis(ax2, 'Original Function')
	else:
		setup_axis(ax1, 'Original Function')
		setup_axis(ax2, 'Wavelet Reconstruction')
	
	# Calculate camera rotation
	if camera == 'rotating':
		angle = frame * 1.5  # Rotation speed
		elevation = 25 + 15 * np.sin(frame * 0.03)  # Elevation variation
	elif camera == 'static_overview':
		angle = 0
		elevation = 90
		
		# Set orthogonal projection
		ax1.set_proj_type('ortho')
		ax2.set_proj_type('ortho')

	# Enhanced lighting setup
	light_angle = frame * 0.05
	light_source_main = plt.matplotlib.colors.LightSource(azdeg=light_angle*180/np.pi, altdeg=60)
	light_source_wavelet = plt.matplotlib.colors.LightSource(azdeg=light_angle*180/np.pi + 90, altdeg=45)
	
	# Determine which axis shows original and which shows reconstruction
	if orientation == 'portrait':
		ax_reconstruction = ax1
		ax_original = ax2
	else:
		ax_reconstruction = ax2
		ax_original = ax1
	
	# Plot original function
	if original_style == 'wireframe':
		surface_orig = ax_original.plot_wireframe(X, Y, f_2d, 
									  color='white', 
									  alpha=original_alpha,
									  linewidth=0.8,
									  rcount=grid_size//4,
									  ccount=grid_size//4)
		# Add semi-transparent surface beneath wireframe
		surface_orig_base = ax_original.plot_surface(X, Y, f_2d, 
										cmap=cmap_original, 
										alpha=original_alpha * 0.7, 
										shade=True,
										antialiased=True,
										linewidth=0,
										rcount=grid_size//2,
										ccount=grid_size//2)
	elif original_style == 'transparent':
		surface_orig = ax_original.plot_surface(X, Y, f_2d, 
								   cmap=cmap_original, 
								   alpha=original_alpha, 
								   shade=True,
								   antialiased=True,
								   linewidth=0,
								   rcount=grid_size//2,
								   ccount=grid_size//2,
								   lightsource=light_source_main)
	elif original_style == 'oscillating_transparent':
		alpha_oscillating = original_alpha + 0.3 * np.sin(frame * original_oscillation_speed)
		alpha_oscillating = np.clip(alpha_oscillating, 0.1, 1.0)
		surface_orig = ax_original.plot_surface(X, Y, f_2d, 
								   cmap=cmap_original, 
								   alpha=alpha_oscillating, 
								   shade=True,
								   antialiased=True,
								   linewidth=0,
								   rcount=grid_size//2,
								   ccount=grid_size//2,
								   lightsource=light_source_main)
	elif original_style == 'none':
		surface_orig = None
	else:  # solid_contrast, gradient, or any other style
		surface_orig = ax_original.plot_surface(X, Y, f_2d, 
								   cmap=cmap_original, 
								   alpha=0.9, 
								   shade=True,
								   antialiased=True,
								   linewidth=0,
								   rcount=grid_size//2,
								   ccount=grid_size//2,
								   lightsource=light_source_main)
	
	# Plot reconstructed surface
	surface_recon = ax_reconstruction.plot_surface(X, Y, f_reconstructed, 
							   cmap=cmap_main, 
							   alpha=0.85, 
							   shade=True,
							   antialiased=True,
							   linewidth=0,
							   rcount=grid_size,
							   ccount=grid_size,
							   lightsource=light_source_main)
	
	# Plot single wavelet contribution on reconstruction plot
	single_wavelet = pywt.waverec2(single_wavelet_coeffs, wavelet_type, mode='periodization')
	if np.any(single_wavelet != 0):
		# Add slight offset to prevent z-fighting
		if camera == 'static_overview':
			# Only plot pixels that are different from zero
			mask = single_wavelet != 0
			if np.any(mask):
				X_filtered = X[mask]
				Y_filtered = Y[mask]
				Z_filtered = np.full_like((single_wavelet)[mask], 10.0)
				color_values = (single_wavelet)[mask]
				
				wavelet_surface = ax_reconstruction.scatter(X_filtered, Y_filtered, Z_filtered,
											  c=color_values, 
											  cmap=cmap_wavelet, 
											  alpha=0.7,
											  s=20)
		else:
			# Only plot pixels that are different from zero
			# Create a mask for pixels with non-zero average of absolute values in their neighborhood
			if single_wavelet_rendering == '2d':
				mask = single_wavelet != 0
				if np.any(mask):
					X_filtered = X[mask]
					Y_filtered = Y[mask]
					Z_filtered = np.full_like((single_wavelet)[mask], 0.01)
				color_values = (single_wavelet)[mask]
				
				wavelet_surface = ax_reconstruction.scatter(X_filtered, Y_filtered, Z_filtered,
											  c=color_values,
											  cmap=cmap_wavelet, 
											  alpha=0.7,
											  s=20)
			else:
				# 3D rendering of single wavelet
				# Add slight offset to prevent z-fighting
				wavelet_surface = ax_reconstruction.plot_surface(X, Y, single_wavelet + 0.01, 
										  cmap=cmap_wavelet, 
										  alpha=0.7, 
										  shade=True,
										  antialiased=True,
										  linewidth=0,
										  rcount=grid_size//2,
										  ccount=grid_size//2,
										  lightsource=light_source_wavelet)

	# Set camera view for both plots (synchronized)
	ax1.view_init(elev=elevation, azim=angle)
	ax2.view_init(elev=elevation, azim=angle)
	
	# Add progress information to the reconstruction plot
	progress_text = f"Wavelets: {wavelets_to_add}/{max_wavelets}"
	ax_reconstruction.text2D(0.02, 0.98, progress_text, transform=ax_reconstruction.transAxes, 
			  color='white', fontsize=12, fontweight='bold',
			  verticalalignment='top',
			  bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
	
	# Update return values based on orientation
	if orientation == 'portrait':
		surface1 = surface_recon
		surface2 = surface_orig
	else:
		surface1 = surface_orig
		surface2 = surface_recon
	
	return surface1, surface2

# Animation
total_frames = max_wavelets * frames_per_wavelet
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=80, blit=False)

# Save the animation with optimized settings
if save != 'none':
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	parameters = f"{layout_name}_{wavelet_type}_{function_type}_{color_scheme}_{max_wavelets}w_{function_seed}s_{frames_per_wavelet}f"
	
	if save.lower() == 'gif':
		filename = f'wavelet2d-{timestamp}-{parameters}.gif'
		print(f"Saving GIF: {filename}")
		print("This may take a while for memory optimization...")
		
		# Use optimized settings for smaller file size and less memory usage
		ani.save(filename, writer='pillow', fps=15, 
				savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1, 'facecolor': 'black'})
		print(f"GIF saved as {filename}")
	
	elif save.lower() == 'mp4':
		filename = f'wavelet2d-{timestamp}-{parameters}.mp4'
		print(f"Saving MP4: {filename}")
		print("This may take a while...")
		
		# Use ffmpeg writer for MP4
		ani.save(filename, writer='ffmpeg', fps=20, bitrate=1800,
				savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1, 'facecolor': 'black'})
		print(f"MP4 saved as {filename}")
	
	else:
		print(f"Unsupported format: {save}. Use 'gif' or 'mp4'")
else:
	print("No file will be saved. Use --save gif or --save mp4 to save animation.")

plt.show()
