from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from data.models import News, User, Community, Comment

@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    pass

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    pass

@admin.register(Community)
class CommunityAdmin(admin.ModelAdmin):
    pass

@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    pass