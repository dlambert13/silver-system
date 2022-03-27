import helpers

"""
image_list = [
    "discrepancy_maps/black.jpg",
    "discrepancy_maps/black.jpg",
    "discrepancy_maps/black.jpg",
    "discrepancy_maps/black.jpg",
    "discrepancy_maps/black.jpg",
    "discrepancy_maps/black.jpg"
]

helpers.stack(image_list, "discrepancy_maps/0_black_column.jpg", direction="column")
"""

image_list_final = [
    "discrepancy_maps/0_black_column.jpg",
    "discrepancy_maps/000_final_4.jpg"
]

helpers.stack(image_list_final, "discrepancy_maps/000_final_5.jpg")
