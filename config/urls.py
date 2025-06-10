from django.contrib import admin
from django.urls import path

from config.views.view import landing_page, signin_page, main_page, plan_choice_page, community_page, anglo_news_page, feedback_page, chat_page, community_all_page, community_write_page, community_detail_page, minitest_page, dev_news_page

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", landing_page),
    path("signin/", signin_page),
    path("main/", main_page),
    path("plan_choice/", plan_choice_page),
    path("community/", community_page),
    path("community_detail/<int:post_id>/", community_detail_page),
    path("community_write/", community_write_page),
    path("community_all/", community_all_page),
    path("anglo_news/", anglo_news_page),
    path("feedback/", feedback_page),
    path("chat/", chat_page),
    path("minitest/", minitest_page),
    path("dev/", dev_news_page)
]
