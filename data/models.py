from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    pass

class News(models.Model):
    news_title = models.CharField(max_length=50)
    news_content = models.CharField(max_length=150)
    news_nation = models.CharField(max_length=10)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    eng_expression = models.CharField(max_length=50)
    korean_expression = models.CharField(max_length=50)
    eng_sentence = models.CharField(max_length=100)
    kor_sentence = models.CharField(max_length=100)

    # 임베딩 해야해서 들어감..
    title_embedding = models.TextField(null=True, blank=True)
    content_embedding = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.news_title

class Community(models.Model):
    post_type = models.CharField(max_length=20, default='community') #community | trend
    post_title = models.CharField(max_length=150)
    post_content = models.TextField()
    like_count = models.IntegerField(default=0)
    comment_count = models.IntegerField(default=0)
    view_count = models.IntegerField(default=0)
    post_created_at = models.DateTimeField(auto_now_add=True)
    post_updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.post_title

class Comment(models.Model):
    community = models.ForeignKey(Community, on_delete=models.CASCADE)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='replies')
    comment_content = models.TextField()
    comment_like_count = models.IntegerField(default=0)
    comment_created_at = models.DateTimeField(auto_now_add=True)
    comment_updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.comment_content[:30]

class Session_feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    teacher_type = models.CharField(max_length=20, default='emma')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    session_status = models.CharField(max_length=10, default='active')
    voca_score = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True)
    gram_score = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True)
    fluence_score = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True)
    completion_score = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True)
    approp_score = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True)
    opic_grade = models.CharField(max_length=5, null=True, blank=True)
    shadowing_sentence1 = models.CharField(max_length=50)
    shadowing_sentence2 = models.CharField(max_length=50)
    shadowing_sentence3 = models.CharField(max_length=50)
    shadowing_sentence4 = models.CharField(max_length=50)
    shadowing_sentence5 = models.CharField(max_length=50)


class Message_feedback(models.Model):
    session_feedback = models.ForeignKey(Session_feedback, on_delete=models.CASCADE)
    message = models.TextField()
    message_at = models.DateTimeField(auto_now_add=True)
    message_corrected = models.TextField(null=True, blank=True)
    message_why = models.TextField(null=True, blank=True)
    message_error = models.BooleanField(default=False)