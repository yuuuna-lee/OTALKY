from django.shortcuts import render, redirect
from data.models import News, Community, Comment
from config.views.news_generator import create_daily_news

def landing_page(request):
    return render(request, 'landing.html')

def minitest_page(request):
    return render(request, 'minitest.html')

def signin_page(request):
    return render(request, 'signin.html')


def main_page(request):
    return render(request, 'main.html')


def community_page(request): #인기글만 보여주는 공간
    posts = Community.objects.filter(post_type='trend')
    context =  {'posts':posts}
    return render(request, 'community.html',context)


def community_all_page(request): #모든 글의 제목들을 보여주는 공간
    posts = Community.objects.all()
    context = {'posts': posts}
    return render(request, 'community_all.html',context)


def community_write_page(request): #글 작성하고, 이를 db에 할당하는 공간
    if request.method == "POST":
        title=request.POST['title']
        content=request.POST['content']
        Community.objects.create(post_title=title, post_content=content)
        return redirect("/community_all/")
    return render(request, 'community_write.html')


def community_detail_page(request, post_id):
    post = Community.objects.get(id=post_id)
    if request.method == 'POST':
        comment_content = request.POST.get('comment_content')
        if comment_content:
            Comment.objects.create(comment_content=comment_content, community=post)
            post.comment_count = post.comment_set.count()
            post.save()
            return redirect(f'/community_detail/{post_id}/')
    comments = Comment.objects.filter(community=post)  # 해당 게시글의 댓글만
    context = {'post': post, 'comments': comments}
    return render(request, 'community_detail.html', context)


def plan_choice_page(request):
    return render(request, 'plan_choice.html')


def anglo_news_page(request):
    nation = request.GET.get('nation')
    keywords = request.GET.get('keyword')
    if nation :
        news = News.objects.filter(news_nation=nation)
    elif keywords:
        news = News.objects.filter(
            news_content__icontains=keywords
        ) | News.objects.filter(
            news_title__icontains=keywords
        ) | News.objects.filter(
            eng_expression__icontains=keywords
        )
    else:
        news = News.objects.all()
    return render(request, 'anglo_news.html', {'news': news})


def chat_page(request):
    return render(request, 'chat.html')


def feedback_page(request):
    return render(request, 'feedback.html')


def dev_news_page(request):
    if request.method == 'POST':
        try:
            create_daily_news()
            message = "뉴스 생성 완료!"
        except:
            message = "생성 실패"
    else:
        message = ""

    news = News.objects.all()[:10]
    return render(request, 'dev_news.html', {'news': news, 'message': message})