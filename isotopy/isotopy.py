#
# isotopy.py - Mathematical Visualization of Isotopy on the Unit Disc
#
# Given two distinct complex numbers z_1 and z_2 inside the unit disc,
#  this script generates an animation of a isotopy between the identity 
# on the unit disc and a homeomorphism that sends z_1 to z_2 and z_2 to z_1
#
# Usage: python isotopy.py x_real x_imag y_real y_imag
# Where:
#  - x_real, x_imag: Real and imaginary parts of the first complex number z_1 (inside unit disc)
#  - y_real, y_imag: Real and imaginary parts of the second complex number z_2 (inside unit disc)
# they must be distinct and strictly inside the unit disc (|z| < 1).
# animation will be saved as 'isotopy_on_disc.gif' and 'isotopy_on_disc.mp4'




import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from matplotlib.animation import PillowWriter, FuncAnimation
import sys

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

def get_coordinates_from_args():
	"""
	Parse command line arguments to get two complex numbers x and y.
	
	Expected format: python isotopy.py x_real x_imag y_real y_imag
	
	Returns:
		tuple: Two complex numbers (x, y) inside the unit disc
		
	Raises:
		SystemExit: If points are outside unit disc or identical
	"""
	if len(sys.argv) >= 5:
		# Parse real and imaginary parts from command line
		x_real = float(sys.argv[1])
		x_imag = float(sys.argv[2])
		y_real = float(sys.argv[3])
		y_imag = float(sys.argv[4])
		
		x_temp = complex(x_real, x_imag)
		y_temp = complex(y_real, y_imag)
		
		# Validate: both points must be strictly inside unit disc
		if abs(x_temp) >= 1 or abs(y_temp) >= 1:
			print("Error: Both points must be in the interior of the unit disc (|z| < 1)")
			sys.exit(1)
		
		# Validate: points must be distinct
		if abs(x_temp - y_temp) < 1e-10:
			print("Error: The two complex numbers must be distinct")
			sys.exit(1)
			
		return x_temp, y_temp
	else:
		# Use default values if no arguments provided
		return -0.7 + 0.3j, 0.2 + 0.2j

# Get the two points we'll work with
x, y = get_coordinates_from_args()

# ============================================================================
# HYPERBOLIC GEOMETRY FUNCTIONS
# ============================================================================

def disc_automorphism(z, b, phi=0):
	"""
	Apply a disc automorphism (Möbius transformation preserving unit disc).
	
	Formula: exp(i*phi) * (z + b) / (conjugate(b)*z + 1)
	
	Args:
		z: Complex number to transform
		b: Complex parameter (|b| < 1)
		phi: Real rotation angle
		
	Returns:
		Complex number after transformation
	"""
	return cmath.exp(1j * phi) * (z + b) / (b.conjugate() * z + 1)

def construct_G_and_inverse(x, y):
	"""
	Construct a hyperbolic isometry G that maps x,y to a,−a on the real axis.
	
	This is done in two steps:
	1. Map x to 0 and apply rotation to make G1(y) real and positive
	2. Apply another automorphism to send 0→a and G1(y)→−a
	
	Args:
		x, y: Two distinct points in the unit disc
		
	Returns:
		tuple: (G, G_inverse, a) where G and G_inverse are functions
			   and a is the real parameter
	"""
	# Step 1: Send x to the origin
	G1_pre = lambda z: (z - x) / (-x.conjugate() * z + 1)
	
	# Find where y maps under G1_pre
	c = G1_pre(y)
	
	# Rotate to make c real and positive
	phi = -cmath.phase(c)
	print(f"Rotation angle phi: {phi}")
	
	G1 = lambda z: cmath.exp(1j * phi) * G1_pre(z)
	c_rot = cmath.exp(1j * phi) * c  # Now real and positive
	c_rot = c_rot.real
	
	print(f"c_rot (should be real and positive): {c_rot}")
	print(f"G1(x) (should be 0): {G1(x)}")
	print(f"G1(y) (should be real): {G1(y)}")

	# Step 2: Find parameter a such that 0→a and c_rot→−a
	# This comes from solving the automorphism equations
	a = (-1 + cmath.sqrt(1 - c_rot**2)) / c_rot
	print(f"Parameter a: {a}")
	
	G2 = lambda z: disc_automorphism(z, a)
	print(f"G2(0) (should equal a): {G2(0)}")
	print(f"G2(c_rot) (should equal -a): {G2(c_rot)}")
	
	# Compose the transformations
	G = lambda z: G2(G1(z))
	
	# Construct inverse transformation (apply inverses in reverse order)
	G2_inv = lambda w: disc_automorphism(w, -a)
	G1_inv = lambda w: disc_automorphism(w * cmath.exp(-1j * phi), x)
	G_inv = lambda z: G1_inv(G2_inv(z))

	return G, G_inv, a.real

# ============================================================================
# HOMOTOPY SETUP
# ============================================================================

# Construct the hyperbolic isometry and its inverse
G, G_inv, a = construct_G_and_inverse(x, y)

# Verify the mapping works correctly
print(f"Verification - x: {x} → G(x): {G(x)} → G_inv(a): {G_inv(a)}")
print(f"Verification - y: {y} → G(y): {G(y)} → G_inv(-a): {G_inv(-a)}")

# Parameters for the homotopy
b = (abs(a) + 1) / 2  # Midpoint between |a| and 1
π = math.pi

def f(r):
	"""
	Radial function that determines rotation amount based on distance from origin.
	
	- For r ≤ |a|: full rotation (π)
	- For |a| < r ≤ b: linear interpolation from π to 0
	- For r > b: no rotation (0)
	
	Args:
		r: Distance from origin (real, non-negative)
		
	Returns:
		Rotation amount in radians
	"""
	if r <= abs(a):
		return π
	elif r <= b:
		# Linear interpolation between π and 0
		return π - (π / (b - abs(a))) * (r - abs(a))
	else:
		return 0.0

def H_t(z, t):
	"""
	Homotopy function that rotates points based on their distance from origin.
	
	Args:
		z: Complex number
		t: Time parameter (0 to 1)
		
	Returns:
		Complex number after applying rotation homotopy
	"""
	return cmath.exp(1j * f(abs(z)) * t) * z

# ============================================================================
# ANIMATION SETUP AND EXECUTION
# ============================================================================

# Grid parameters for visualization
angles_rays = np.linspace(0, 2*np.pi, 13)      # 13 radial rays
angles_circles = np.linspace(0, 2*np.pi, 200)  # 200 points per circle (smooth)
radii_rays = np.linspace(0.05, 1.0, 300)       # 300 points per ray (smooth)
radii_circles = np.linspace(0.05, 1.0, 8)      # 8 concentric circles
radii_circles = np.append(radii_circles, b)  # Include radius b in the circles

# Set up the figure
fig, ax = plt.subplots(figsize=(3, 3))

# Time values for animation (from 0 to 1)
t_values = np.linspace(0, 1, 51)  # 51 frames

def update(t):
	"""
	Update function for animation. Draws the transformed grid at time t.
	
	Args:
		t: Current time parameter (0 to 1)
	"""
	ax.clear()
	ax.set_title(f"t = {t:.2f}")
	
	# Draw radial rays
	for θ in angles_rays:
		# Generate points along a ray from center to boundary
		zs = radii_rays * np.exp(1j * θ)
		# Apply homotopy and inverse transformation
		img = [G_inv(H_t(z, t)) for z in zs]
		# Plot the transformed ray
		# Use different colors for each ray
		colors = plt.cm.rainbow(np.linspace(0, 1, len(angles_rays)))
		ray_color = colors[list(angles_rays).index(θ)]
		ax.plot([z.real for z in img], [z.imag for z in img], 
				linewidth=2, color=ray_color)
	
	# Draw concentric circles
	for r in radii_circles:
		# Generate points along a circle of radius r
		zs = r * np.exp(1j * angles_circles)
		# Apply homotopy and inverse transformation
		img = [G_inv(H_t(z, t)) for z in zs]
		# Plot the transformed circle
		ax.plot([z.real for z in img], [z.imag for z in img], 
				linewidth=1, color='blue')
	
	# Mark the original points x and y (transformed)
	pts = [-a, a]  # Points in the G-transformed space
	img_pts = [G_inv(H_t(p, t)) for p in pts]
	ax.scatter([p.real for p in img_pts], [p.imag for p in img_pts], 
			   color='black', s=50, zorder=5)
	
	# Set up the plot appearance
	ax.set_aspect('equal')
	ax.set_xlim(-1.05, 1.05)
	ax.set_ylim(-1.05, 1.05)
	ax.axis('off')  # Hide axes for cleaner look

# Create and save animation
print("Creating animation...")
ani = FuncAnimation(fig, update, frames=t_values, interval=100)
ani.save("isotopy_on_disc.gif", writer=PillowWriter(fps=10))
ani.save("isotopy_on_disc.mp4", writer='ffmpeg', fps=10)
print("Animation saved as 'isotopy_on_disc.gif' and 'isotopy_on_disc.mp4'")

# Display the final result
plt.show()
