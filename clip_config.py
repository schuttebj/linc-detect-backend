"""
CLIP Configuration for Testing Different Models and Approaches

This file allows you to quickly switch between different CLIP models
and classification approaches without modifying the core code.

Edit the values below and restart the backend to test different configurations.
"""

# ============================================================================
# CLIP MODEL SELECTION
# ============================================================================
# Choose which CLIP model to use:
# - "openai/clip-vit-base-patch32"   → Faster, less memory (~350MB), good accuracy
# - "openai/clip-vit-large-patch14"  → Slower, more memory (~890MB), better accuracy

# CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_MODEL = "openai/clip-vit-large-patch14"  # Testing large model

# ============================================================================
# CLASSIFICATION LABELS
# ============================================================================
# Simplified to 8 clear categories based on the original list:
# - car, pickup, SUV, van, bus, semitruck, motorcycle, trailer
#
# Van refinement (optional second-round):
# - If CLIP identifies "van", system can refine to: minivan, small commercial van, large commercial van
# - This helps distinguish Class 1 (light van/minivan) from Class 2 (heavy commercial van)

# ============================================================================
# MODEL DETAILS
# ============================================================================
# Base Model (clip-vit-base-patch32):
#   - Inference time: ~150-200ms per image
#   - Memory: ~350MB
#   - Accuracy: Good for most cases
#   - Best for: Production with limited resources
#
# Large Model (clip-vit-large-patch14):
#   - Inference time: ~300-500ms per image
#   - Memory: ~890MB
#   - Accuracy: Better, especially for edge cases
#   - Best for: High accuracy requirements, sufficient resources
#
# ============================================================================
# APPROACH DETAILS
# ============================================================================
# Detailed Approach (11 labels):
#   - car, pickup truck, SUV, minivan, motorcycle
#   - delivery truck, bus
#   - semi truck, truck with trailer
#   - trailer
#   Pro: More specific classification
#   Con: Can confuse similar types (SUV vs pickup)
#
# Binary Approach (8 labels):
#   - Light: light passenger vehicle, private car, light truck, motorcycle
#   - Heavy: heavy commercial vehicle, large truck, commercial bus, articulated lorry
#   Pro: Clear light vs heavy distinction
#   Con: Less specific vehicle types, relies on axle count
#
# ============================================================================
# TESTING WORKFLOW
# ============================================================================
# 1. Edit CLIP_MODEL above to switch between base/large
# 2. Edit USE_BINARY_APPROACH to switch between detailed/binary
# 3. Restart backend (Railway will auto-deploy on git push)
# 4. Frontend will automatically show BOTH approaches side-by-side for comparison
# 5. The "Active" badge shows which approach is used for final classification
# 6. Compare inference times and accuracy in the debug panel

