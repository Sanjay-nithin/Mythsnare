from django.db import models

class Register(models.Model):
    username = models.CharField(max_length=100, unique=True)  # Updated to username
    phone_number = models.CharField(max_length=15, unique=True)  # Phone number
    password = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.username} ({self.phone_number})"

class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.email}"
    
class NewsArticle(models.Model):
    title = models.CharField(max_length=255)
    summary = models.TextField()
    link = models.URLField()
    content_embedding = models.BinaryField()  # Store vector embeddings here

    def __str__(self):
        return self.title
