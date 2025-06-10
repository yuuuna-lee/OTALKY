import os
import requests
import json
from openai import OpenAI
import numpy as np
import re
from datetime import datetime, timedelta
from data.models import News
import time
from difflib import SequenceMatcher

# API 키들
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


def remove_stopwords(text):
    """간단한 불용어 제거"""
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are',
                 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
    words = text.lower().split()
    return [word for word in words if word not in stopwords and len(word) > 2]


def check_duplicate_title(new_title):
    """개선된 제목 유사도 체크 (불용어 제거 + Jaccard + Levenshtein)"""
    recent_news = News.objects.filter(
        uploaded_at__gte=datetime.now() - timedelta(days=30)
    )

    # 불용어 제거된 단어셋
    new_words = set(remove_stopwords(new_title))

    for news in recent_news:
        old_words = set(remove_stopwords(news.news_title))

        if new_words and old_words:
            # Jaccard 유사도 (불용어 제거 후)
            jaccard = len(new_words & old_words) / len(new_words | old_words)

            # Levenshtein 유사도
            levenshtein = SequenceMatcher(None, new_title.lower(), news.news_title.lower()).ratio()

            # 둘 중 하나라도 임계값 넘으면 중복
            if jaccard > 0.5 or levenshtein > 0.7:
                print(f"제목 중복 감지: {news.news_title[:30]}... (J:{jaccard:.2f}, L:{levenshtein:.2f})")
                return True
    return False


def determine_news_nation(title, url="", source=""):
    """제목, URL, 소스 기반으로 뉴스 국가 추정"""

    # URL 기반 판단 (가장 정확)
    if url:
        url_lower = url.lower()
        if any(domain in url_lower for domain in ['cnn.com', 'nytimes.com', 'wsj.com', 'usatoday.com']):
            return "US"
        elif any(domain in url_lower for domain in ['bbc.com', 'guardian.com', 'independent.co.uk', 'telegraph.co.uk']):
            return "UK"
        elif any(domain in url_lower for domain in ['abc.net.au', 'news.com.au', 'theaustralian.com.au']):
            return "AU"

    # 소스 기반 판단
    if source:
        source_lower = source.lower()
        if any(s in source_lower for s in ['cnn', 'nytimes', 'usa', 'american']):
            return "US"
        elif any(s in source_lower for s in ['bbc', 'guardian', 'uk', 'british', 'london']):
            return "UK"
        elif any(s in source_lower for s in ['australia', 'sydney', 'melbourne']):
            return "AU"

    # 제목 내용 기반 판단 (키워드)
    title_lower = title.lower()
    us_keywords = ['washington', 'biden', 'trump', 'congress', 'senate', 'white house', 'pentagon', 'fbi', 'nasa',
                   'wall street']
    uk_keywords = ['london', 'westminster', 'downing street', 'parliament', 'british', 'england', 'scotland', 'wales']
    au_keywords = ['sydney', 'melbourne', 'canberra', 'australian', 'australia']

    us_score = sum(1 for keyword in us_keywords if keyword in title_lower)
    uk_score = sum(1 for keyword in uk_keywords if keyword in title_lower)
    au_score = sum(1 for keyword in au_keywords if keyword in title_lower)

    if us_score > uk_score and us_score > au_score:
        return "US"
    elif uk_score > au_score:
        return "UK"
    elif au_score > 0:
        return "AU"

    # 기본값 (미국이 뉴스 비중이 높음)
    return "US"


def optimize_embedding(embedding):
    """임베딩 최적화 (float16으로 압축)"""
    if embedding:
        try:
            # numpy array로 변환 후 float16으로 압축
            embedding_array = np.array(embedding, dtype=np.float16)
            # 다시 리스트로 변환하여 JSON 직렬화 가능하게
            return embedding_array.astype(float).tolist()
        except Exception as e:
            print(f"임베딩 최적화 오류: {e}")
            return embedding
    return None

def cosine_similarity(a, b):
    """정확한 코사인 유사도 계산"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_duplicate_expression(expression):
    """개선된 영어 표현 중복 체크 (의미 기반)"""
    used_expressions = News.objects.filter(
        uploaded_at__gte=datetime.now() - timedelta(days=30)
    ).values_list('eng_expression', flat=True)

    # 정확히 같은 표현
    if expression.lower() in [expr.lower() for expr in used_expressions]:
        print(f"표현 정확 중복: {expression}")
        return True

    # 임베딩 기반 의미 유사도 체크
    try:
        new_embedding = get_embedding(expression)
        if not new_embedding:
            return False

        for used_expr in used_expressions:
            used_embedding = get_embedding(used_expr)
            if used_embedding:
                similarity = cosine_similarity(new_embedding, used_embedding)
                if similarity > 0.85:  # 의미가 매우 유사하면 중복
                    print(f"표현 의미 중복: {expression} ≈ {used_expr} (유사도: {similarity:.2f})")
                    return True
    except Exception as e:
        print(f"표현 유사도 체크 오류: {e}")

    return False


def get_embedding(text):
    """텍스트 임베딩 생성 (오류 처리 개선)"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None


def check_content_similarity(title, content):
    """개선된 임베딩 기반 내용 유사도 체크"""
    new_text = f"{title} {content}"
    new_embedding = get_embedding(new_text)

    if not new_embedding:
        return False

    recent_news = News.objects.filter(
        uploaded_at__gte=datetime.now() - timedelta(days=30)
    ).exclude(title_embedding__isnull=True)

    for news in recent_news:
        if news.title_embedding:
            try:
                old_embedding = json.loads(news.title_embedding)
                # 정확한 코사인 유사도 계산
                similarity = cosine_similarity(new_embedding, old_embedding)
                if similarity > 0.8:
                    return True
            except Exception as e:
                print(f"유사도 계산 오류: {e}")
                continue

    return False


def get_news_data():
    """뉴스 + Reddit 데이터 수집"""
    all_candidates = []

    # NewsAPI
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'apiKey': NEWS_API_KEY,
            'q': 'breaking OR latest OR news OR update',
            'language': 'en',
            'domains': 'bbc.com,cnn.com,theguardian.com,reuters.com,techcrunch.com,engadget.com,politico.com,wsj.com,nytimes.com,washingtonpost.com,independent.co.uk,abc.net.au',
            'pageSize': 30,
            'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params)
        articles = response.json().get('articles', [])

        for article in articles:
            if article.get('title') and article.get('description'):
                all_candidates.append({
                    'title': article['title'],
                    'content': article['description'],
                    'source': 'news'
                })
    except Exception as e:
        print(f"NewsAPI 오류: {e}")

    # Reddit
    subreddits = ['worldnews', 'technology', 'entertainment', 'politics']
    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=8"
            response = requests.get(url, headers={'User-Agent': 'NewsBot/1.0'})
            data = response.json()

            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                if title and len(title) > 20:  # 너무 짧은 제목 제외
                    all_candidates.append({
                        'title': title,
                        'content': title,
                        'source': 'reddit'
                    })
            time.sleep(1)
        except Exception as e:
            print(f"Reddit 오류 ({sub}): {e}")

    return all_candidates


def fact_check(title):
    """강화된 Serper 팩트체크 + GPT 보조 검증"""
    try:
        url = "https://google.serper.dev/search"
        headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

        response = requests.post(url, json={'q': title, 'num': 5}, headers=headers)
        results = response.json()

        reliable_sources = ['bbc', 'cnn', 'reuters', 'guardian', 'ap', 'nytimes', 'wsj']
        reliable_count = 0
        title_mentions = 0

        for result in results.get('organic', []):
            link = result.get('link', '').lower()
            snippet = result.get('snippet', '').lower()
            result_title = result.get('title', '').lower()

            # 신뢰할 만한 소스 체크
            if any(source in link for source in reliable_sources):
                reliable_count += 1

                # 제목이나 스니펫에 주요 키워드 포함 여부 체크
                title_words = set(title.lower().split()[:3])  # 제목 앞 3단어
                if any(word in snippet or word in result_title for word in title_words):
                    title_mentions += 1

        # 기본 신뢰도: 신뢰할 만한 소스 2개 이상 + 관련 언급 1개 이상
        if reliable_count >= 2 and title_mentions >= 1:
            return True

        # GPT 보조 팩트체크 (의심스러운 경우만)
        if reliable_count < 2:
            gpt_check = gpt_fact_check(title)
            return gpt_check

        return reliable_count >= 1

    except Exception as e:
        print(f"팩트체크 오류: {e}")
        return True  # 오류시 일단 통과


def gpt_fact_check(title):
    """GPT 보조 팩트체크"""
    try:
        prompt = f"""
다음 뉴스 제목이 거짓뉴스일 가능성을 평가해주세요:
"{title}"

평가 기준:
- 상식적으로 말이 되는가?
- 과장이나 선정적 표현이 있는가?
- 검증 가능한 사실인가?

답변: PASS (신뢰할만함) 또는 FAIL (의심스러움)
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip().upper()
        return "PASS" in result

    except Exception as e:
        print(f"GPT 팩트체크 오류: {e}")
        return True


def calculate_interest_score(title, content):
    """GPT로 한국인 관심도 평가 (백업 로직 강화)"""
    text = f"{title} {content}"

    prompt = f"""
다음 뉴스가 한국 20-30대에게 얼마나 흥미로울지 0-100점으로 평가하세요.

뉴스: {text[:200]}

평가 기준:
- 한국과의 연관성
- 20-30대 관심사 (테크, 연예, 경제, 사회이슈)
- 화제성/화제가 될 만한 정도
- 교육적 가치

숫자만 답변: (예: 75)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.3
        )

        score_text = response.choices[0].message.content.strip()
        score = int(''.join(filter(str.isdigit, score_text)))
        return min(max(score, 0), 100)

    except Exception as e:
        print(f"관심도 평가 오류: {e}")
        # 강화된 백업 로직 (키워드별 가중치)
        keyword_weights = {
            'technology': 20, 'ai': 25, 'celebrity': 15, 'politics': 10,
            'economy': 15, 'climate': 12, 'korea': 30, 'samsung': 20,
            'netflix': 18, 'tesla': 22, 'apple': 20, 'bitcoin': 25
        }

        text_lower = text.lower()
        backup_score = sum(weight for keyword, weight in keyword_weights.items()
                           if keyword in text_lower)
        return min(backup_score, 80)


def generate_korean_news(title, content, url="", source_name=""):
    """GPT로 한국 뉴스 생성 (국가 자동 추정)"""
    used_expressions = list(News.objects.filter(
        uploaded_at__gte=datetime.now() - timedelta(days=30)
    ).values_list('eng_expression', flat=True))

    # 뉴스 국가 미리 추정
    predicted_nation = determine_news_nation(title, url, source_name)

    prompt = f"""
    다음 영어 뉴스를 한국 20-30대가 흥미로워할 뉴스로 재작성해주세요.

    제목: {title}
    내용: {content}
    예상 국가: {predicted_nation}

    요구사항:
    1. 제목 45자 이내, 내용 150자 이내
    2. 구체적인 사실과 숫자 포함 (누가, 언제, 어떻게, 왜)
    3. 자연스러운 20-30대 문체 ("~했어요", "~네요", "~죠" 사용)
    4. "~합니다", "~입니다" 금지
    5. 일상회화 영어표현 우선
    6. 한국과의 연관성은 자연스러운 경우에만 (억지로 끼워넣지 말 것)
    7. 제목은 간단하지만 내용파악이 쉽고, 가독성 있게 작성
    8. 번역체 금지, authentic한 한국어로 content 작성하기

    문체 예시:
    ❌ "CEO는 발표했습니다" 
    ✅ "CEO가 깜짝 발표했어요"
    ❌ "기술이 발전하고 있습니다"
    ✅ "기술이 엄청 발전하고 있네요"

    사용금지 표현: {', '.join(used_expressions[-10:])}
    "영어 표현은 1-3단어로 간단하게 작성하시오.
    "긴 구문이나 문장 금지"
    초등학생도 알법한 쉬운 영어 표현은 제공하지 않는다.

    반드시 아래 JSON 형식으로만 답변:
    {{
        "news_title": "한국어 제목 (45자 이내)",
        "news_content": "한국어 내용 (150자 이내, 구체적 사실 포함)", 
        "news_nation": "{predicted_nation}",
        "eng_expression": "영어표현",
        "korean_expression": "한국어 뜻",
        "eng_sentence": "영어 예문",
        "kor_sentence": "한국어 번역"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "한국 20-30대용 영어학습 뉴스 작성 전문가. 반드시 JSON 형식으로만 답변하세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        content = response.choices[0].message.content.strip()

        # JSON 추출 (정규식 사용으로 신뢰성 강화)
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                korean_news = json.loads(json_text)

                # 필수 필드 검증
                required_fields = ['news_title', 'news_content', 'news_nation', 'eng_expression', 'korean_expression',
                                   'eng_sentence', 'kor_sentence']
                if all(field in korean_news for field in required_fields):
                    # news_nation 값 검증 및 보정
                    if korean_news['news_nation'] not in ['US', 'UK', 'AU']:
                        korean_news['news_nation'] = predicted_nation
                        print(f"국가 보정: {korean_news['news_nation']} → {predicted_nation}")

                    # 생성된 뉴스 검수
                    if quality_check(korean_news):
                        return korean_news
                    else:
                        print("품질 검수 실패")
                        return None
                else:
                    print(f"필수 필드 누락: {korean_news}")
                    return None
            else:
                print("JSON 형식을 찾을 수 없음")
                return None

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"응답 내용: {content[:100]}...")
            return None

    except Exception as e:
        print(f"뉴스 생성 오류: {e}")
        return None


def quality_check(korean_news):
    """생성된 뉴스 품질 검수"""
    try:
        prompt = f"""
다음 뉴스에 오타나 이상한 문장은 없는지 검수해주세요:

제목: {korean_news['news_title']}
내용: {korean_news['news_content']}
영어표현: {korean_news['eng_expression']} - {korean_news['korean_expression']}
예문: {korean_news['eng_sentence']} / {korean_news['kor_sentence']}

문제없으면 "PASS", 문제있으면 "FAIL"로 답변
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip().upper()
        return "PASS" in result

    except Exception as e:
        print(f"품질 검수 오류: {e}")
        return True  # 오류시 통과


def create_daily_news():
    """메인 뉴스 생성 함수"""
    print("🚀 뉴스 생성 시작")

    # 오늘 뉴스 개수 체크
    today_count = News.objects.filter(
        uploaded_at__date=datetime.now().date()
    ).count()

    if today_count >= 8:
        print("✅ 오늘 뉴스 8개 완료")
        return

    need_count = 8 - today_count
    print(f"📝 {need_count}개 뉴스 필요")

    # 1. 데이터 수집
    candidates = get_news_data()
    print(f"📊 {len(candidates)}개 후보 수집")

    # 2. 관심도 점수로 정렬
    scored_candidates = []
    for candidate in candidates:
        if not check_duplicate_title(candidate['title']):
            score = calculate_interest_score(candidate['title'], candidate['content'])
            if score >= 20:
                candidate['score'] = score
                scored_candidates.append(candidate)

    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    print(f"✅ {len(scored_candidates)}개 후보 선별")

    # 3. 뉴스 생성
    created_count = 0

    for candidate in scored_candidates[:need_count * 3]:  # 여유분
        if created_count >= need_count:
            break

        # 팩트체크
        if not fact_check(candidate['title']):
            print(f"❌ 팩트체크 실패: {candidate['title'][:30]}")
            continue

        # 임베딩 유사도 체크
        if check_content_similarity(candidate['title'], candidate['content']):
            print(f"❌ 내용 유사: {candidate['title'][:30]}")
            continue

        # GPT 생성
        korean_news = generate_korean_news(
            candidate['title'],
            candidate['content'],
            candidate.get('url', ''),
            candidate.get('source_name', '')
        )
        if not korean_news:
            continue

        # 영어표현 중복 체크
        if check_duplicate_expression(korean_news['eng_expression']):
            print(f"❌ 표현 중복: {korean_news['eng_expression']}")
            continue

        # 임베딩 생성 및 최적화
        new_text = f"{korean_news['news_title']} {korean_news['news_content']}"
        embedding = get_embedding(new_text)
        optimized_embedding = optimize_embedding(embedding)

        try:
            news_obj = News.objects.create(
                news_title=korean_news['news_title'][:50],
                news_content=korean_news['news_content'][:150],
                news_nation=korean_news['news_nation'],
                eng_expression=korean_news['eng_expression'][:50],
                korean_expression=korean_news['korean_expression'][:50],
                eng_sentence=korean_news['eng_sentence'][:100],
                kor_sentence=korean_news['kor_sentence'][:100],
                title_embedding=json.dumps(optimized_embedding) if optimized_embedding else None
            )

            created_count += 1
            print(f"✅ 뉴스 저장 완료 ({created_count}/{need_count}): {korean_news['news_title'][:30]}...")
            print(
                f"   🌍 국가: {korean_news['news_nation']} | 📝 표현: {korean_news['eng_expression']} - {korean_news['korean_expression']}")

            # 임베딩 압축 효과 로깅
            if embedding and optimized_embedding:
                original_size = len(json.dumps(embedding))
                compressed_size = len(json.dumps(optimized_embedding))
                compression_ratio = (1 - compressed_size / original_size) * 100
                print(f"   💾 임베딩 압축: {original_size}→{compressed_size} bytes ({compression_ratio:.1f}% 절약)")

        except Exception as e:
            print(f"❌ DB 저장 오류: {e}")
            print(f"   뉴스 데이터: {korean_news}")

        time.sleep(2)

    print(f"🎉 총 {created_count}개 뉴스 생성 완료!")