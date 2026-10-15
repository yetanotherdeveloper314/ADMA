"""
Dataset registry and label normalization for ADMA.

Add new datasets by appending an entry to DATASETS.
Adjust LABEL_MAP to control how raw dataset labels are normalized
(or set to None to drop non-military classes).
"""

DATASETS: dict[str, dict] = {
    "tank_images": {
        "workspace": "cv-nuyii",
        "project": "tank-images",
        "version": 6,
        "location": "data/tank_images",
        "description": "Tank Images (6.7K images)",
    },
    "military_vehicles": {
        "workspace": "oleksandr-tara",
        "project": "military-vehicle-detection-juleg",
        "version": 7,
        "location": "data/military_vehicles",
        "description": "Military Vehicle Detection (3.1K images)",
    },
    "military_vehicles_obj": {
        "workspace": "military-vehicle-object-detection",
        "project": "military-vehicles-object-detection",
        "version": 16,
        "location": "data/military_vehicles_obj",
        "description": "Military Vehicles Object Detection (2.1K images)",
    },
    "military_objects": {
        "workspace": "military-object-detection",
        "project": "military-object-detection-eay0m",
        "version": 5,
        "location": "data/military_objects",
        "description": "Military Object Detection (2.3K images)",
    },
}

LABEL_MAP: dict[str, str | None] = {
    # Tank Images
    "person": None,
    "tank": "tank",
    "hong-di-chuyen": None,
    "hong-hoa-luc": None,
    "hong-hoan-toan": None,
    # Military Vehicle Detection
    # This dataset will be excluded from combined training.
    "Vehicle": None,
    # Military Vehicles Object Detection
    "civ_hel": None,
    "drone": "drone",
    "jet": "military_aircraft",
    "land": None,
    "large_mil_plane": "military_aircraft",
    "mil_helicopter": "military_helicopter",
    "stealth": "military_aircraft",
    "tech_vehicle": None,
    # Military Object Detection
    "Aircraft": "military_aircraft",
    "Drone": "drone",
    "Helicopter": "military_helicopter",
    "Missile": "missile",
    "Missile Launchers": "missile_launcher",
    "Person": None,
    "Tank": "tank",
    "Truck": "military_truck",
    "Warship": "warship",
    "Weapons": None,
}

COMBINED_DATASET_EXCLUSIONS: dict[str, str] = {
    "military_vehicles": (
        "The generic Vehicle class cannot be reliably mapped to the "
        "approved taxonomy without verifying or relabeling annotations."
    ),
}
