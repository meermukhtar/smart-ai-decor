
from django.db import models


class Room(models.Model):

    ROOM_TYPES = [
        ("living_room", "Living Room"),
        ("bedroom", "Bedroom"),
        ("kitchen", "Kitchen"),
        ("dining_room", "Dining Room"),
        ("office", "Office"),
        ("gym", "Home Gym"),
        ("reading_nook", "Reading Nook"),
        ("other", "Other"),
    ]

    name = models.CharField(max_length=200, default="Room")

    image = models.ImageField(
        upload_to="rooms/"
    )

    decorated_image = models.ImageField(
        upload_to="decorated/",
        blank=True,
        null=True
    )

    annotated_image = models.ImageField(
        upload_to="annotated/",
        blank=True,
        null=True
    )

    depth_map = models.ImageField(
        upload_to="depth/",
        blank=True,
        null=True
    )

    point_cloud = models.FileField(
        upload_to="3d/",
        blank=True,
        null=True
    )

    room_type = models.CharField(
        max_length=50,
        choices=ROOM_TYPES,
        blank=True,
        null=True
    )

    analysis = models.JSONField(
        blank=True,
        null=True
    )

    suggestions = models.JSONField(
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.name