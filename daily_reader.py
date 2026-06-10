#!/usr/bin/env python3
"""
Daily English Reader
每日自动抓取英文新闻 → AI筛选 → 翻译排版 → 发送到QQ邮箱
"""

import os
import re
import json
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date

import requests
import feedparser
from bs4 import BeautifulSoup
from readability import Document
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv(Path(__file__).parent / ".env")

# ── 配置 ─────────────────────────────────────────────

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465
SENDER_EMAIL = os.environ["SENDER_EMAIL"]
SENDER_PASSWORD = os.environ["SENDER_PASSWORD"]  # QQ邮箱SMTP授权码
RECEIVER_EMAIL = os.environ["RECEIVER_EMAIL"]  # 多个收件人用英文逗号分隔
HTTP_PROXY = os.environ.get("HTTP_PROXY", "")  # Clash 默认 http://127.0.0.1:7897

# 只给外网爬取用代理，不影响 DeepSeek API
http_session = requests.Session()
if HTTP_PROXY:
    http_session.proxies = {"http": HTTP_PROXY, "https": HTTP_PROXY}
    print(f"[Proxy] 爬虫走代理 {HTTP_PROXY}")
else:
    print("[Proxy] 未配置代理，直连")

# 备用 session：跳过 SSL 验证，用于处理 Clash 对某些网站的 SSL 兼容问题
noverify_session = requests.Session()
if HTTP_PROXY:
    noverify_session.proxies = {"http": HTTP_PROXY, "https": HTTP_PROXY}
noverify_session.verify = False

INTERESTS = "金融/科技/公共治理/政治/社会经济"
MAX_ARTICLES = 3

RSS_FEEDS = [
    ("https://feeds.bbci.co.uk/news/rss.xml", "BBC News"),
    ("https://feeds.bbci.co.uk/news/business/rss.xml", "BBC Business"),
    ("https://feeds.bbci.co.uk/news/technology/rss.xml", "BBC Technology"),
    ("https://feeds.bbci.co.uk/news/politics/rss.xml", "BBC Politics"),
    ("https://www.theguardian.com/world/rss", "The Guardian World"),
    ("https://www.theguardian.com/business/rss", "The Guardian Business"),
    ("https://www.theguardian.com/technology/rss", "The Guardian Technology"),
    ("https://www.theguardian.com/politics/rss", "The Guardian Politics"),
]

# DeepSeek API 不走代理，直连
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ── 步骤1：抓取 RSS ──────────────────────────────────

def strip_html(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _get_url(url, timeout=20):
    """抓取 URL，先尝试正常 SSL，失败则跳过 SSL 验证重试"""
    for attempt, session in enumerate([http_session, noverify_session]):
        try:
            resp = session.get(url, headers={"User-Agent": UA}, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == 0:
                continue  # 第一次失败，用 noverify 重试
            raise e


def fetch_articles():
    articles = []
    seen = set()

    for url, source_name in RSS_FEEDS:
        try:
            print(f"  Fetching {source_name}...", flush=True)
            resp = _get_url(url, timeout=20)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)

                summary = strip_html(entry.get("summary", "") or entry.get("description", ""))
                link = entry.get("link", "")

                if not link:
                    continue

                articles.append({
                    "title": title,
                    "summary": summary[:500],
                    "link": link,
                    "source": source_name,
                })
            print(f"  [OK] {source_name}: {len(feed.entries)} entries")
        except Exception as e:
            print(f"  [FAIL] {source_name}: {e}")

    print(f"  Total unique articles: {len(articles)}")
    return articles


# ── 步骤2：AI 筛选 ────────────────────────────────────

FILTER_PROMPT = """你是一位资深新闻编辑。以下是从各大英文媒体抓取的文章标题和摘要。

我的兴趣方向：{interests}

请从中选出最值得阅读的 {max_n} 篇文章。要求：
- 与我的兴趣方向高度相关
- 是今天的重要新闻，不是软文或花边
- 尽量覆盖不同主题，不要全选同一方向

返回纯 JSON 数组（不要 markdown 代码块），每个元素：
{{"index": 数字(从1开始), "reason": "一句话推荐理由(中文)"}}

文章列表：
{article_list}"""


def ai_filter(articles):
    if len(articles) <= MAX_ARTICLES:
        return articles

    # 取前 40 篇给 AI 选，省钱
    candidates = articles[:40]
    article_list = "\n".join(
        f"{i+1}. [{a['source']}] {a['title']}\n   摘要: {a['summary'][:200]}"
        for i, a in enumerate(candidates)
    )

    prompt = FILTER_PROMPT.format(
        interests=INTERESTS,
        max_n=MAX_ARTICLES,
        article_list=article_list,
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位资深新闻编辑。只返回纯 JSON 数组，不要 markdown 代码块。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        selected = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  AI filter JSON parse failed, falling back to top {MAX_ARTICLES}")
        return candidates[:MAX_ARTICLES]

    result = []
    for item in selected[:MAX_ARTICLES]:
        idx = int(item["index"]) - 1
        if 0 <= idx < len(candidates):
            a = candidates[idx].copy()
            a["ai_reason"] = item.get("reason", "")
            result.append(a)

    return result


# ── 步骤3：爬取正文 ───────────────────────────────────

def fetch_body(url):
    """多策略提取正文"""
    try:
        resp = _get_url(url, timeout=15)
        html = resp.text

        # 策略1：readability 提取
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text_r = soup.get_text()
        text_r = re.sub(r"\n{3,}", "\n\n", text_r).strip()

        # 策略2：直接提取所有 <p> 段落（readability 失败时兜底）
        soup_full = BeautifulSoup(html, "html.parser")
        # 移除 script/style/nav 等噪音标签
        for tag in soup_full(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paragraphs = [p.get_text().strip() for p in soup_full.find_all("p")]
        text_p = "\n\n".join(p for p in paragraphs if len(p) > 40)

        # 取较长者
        best = text_r if len(text_r) > len(text_p) else text_p

        if len(best) < 200:
            print(f"    正文太短 ({len(best)} chars)，可能被付费墙或JS阻挡")
            return ""

        print(f"    提取到 {len(best)} chars")
        return best[:6000]

    except Exception as e:
        print(f"    提取失败: {e}")
        return ""


# ── 步骤4：AI 翻译排版 ────────────────────────────────

TRANSLATE_PROMPT = """你是一位英语学习导师。请处理以下英文新闻，生成一份学习材料。读者水平为雅思阅读7.5分，英文功底扎实。

===== 要求 =====
1. **段落级中英对照**：按文章自然段落拆分，每段英文原文后紧跟中文翻译。保持原文段落结构，不要合并或跳过段落。
2. **重点词汇**：只选取 5-8 个对雅思7.5水平读者也有挑战的词汇/短语，标准：
   - GRE/SAT 级别的进阶词汇
   - 地道习语、固定搭配、熟词僻义
   - 金融/科技/政治领域的专业术语
   - 切勿选高中或四级难度的基础词汇（如 government, economy, technology 等）

===== 输出格式（严格按此 Markdown） =====

## [文章标题]

**来源：[来源名称]** | **推荐理由：[理由]**

---

### 段落精读

**Paragraph 1**

[英文原文段落1]

[中文翻译段落1]

**Paragraph 2**

[英文原文段落2]

[中文翻译段落2]

（每个自然段都这样处理，不要遗漏）

---

### 进阶词汇

| 单词/短语 | 释义 | 原文例句 |
|----------|------|---------|
| nascent | 新生的，萌芽期的 | The nascent industry faces regulatory hurdles. |
| ... | ... | ... |

===== 要处理的文章 =====

标题：{title}
来源：{source}
推荐理由：{reason}

原文：
{body}"""


def ai_translate(article):
    body = article["body"]
    if not body:
        return None

    prompt = TRANSLATE_PROMPT.format(
        title=article["title"],
        source=article["source"],
        reason=article.get("ai_reason", "今日精选"),
        body=body,
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位专业英语学习导师。请严格按照要求的格式输出。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=8000,
    )

    return resp.choices[0].message.content


# ── 步骤5：Markdown → HTML ───────────────────────────

def md_to_html(md):
    """将 Markdown 转为适合邮件的 HTML"""
    lines = md.split("\n")
    out = []
    in_table = False
    table_lines = []
    in_hr = False

    for line in lines:
        stripped = line.strip()

        # 空行
        if not stripped:
            if in_table:
                out.append(render_table(table_lines))
                table_lines = []
                in_table = False
            out.append("<br>")
            continue

        # 表格行
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(stripped)
            continue

        # 刚退出表格
        if in_table:
            out.append(render_table(table_lines))
            table_lines = []
            in_table = False

        # 分割线
        if stripped.startswith("---"):
            out.append('<hr style="border:none;border-top:1px solid #ddd;margin:20px 0">')
            continue

        # 标题
        if stripped.startswith("## "):
            out.append(f'<h2 style="color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:8px">{stripped[3:]}</h2>')
            continue
        if stripped.startswith("### "):
            out.append(f'<h3 style="color:#34495e;margin-top:20px">{stripped[4:]}</h3>')
            continue

        # 加粗段落（如 **来源**）
        if stripped.startswith("**"):
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", stripped)
            out.append(f'<p style="color:#7f8c8d;font-size:14px">{text}</p>')
            continue

        # 段落标题（Paragraph N）
        if re.match(r"^\*\*Paragraph \d+\*\*$", stripped):
            num = re.search(r"\d+", stripped).group()
            out.append(f'<p style="font-weight:bold;color:#3498db;margin-top:15px"> Paragraph {num}</p>')
            continue

        # 普通段落 - 尝试区分英文和中文段落
        cls = detect_paragraph_class(stripped)
        out.append(f'<p class="{cls}">{stripped}</p>')

    if in_table and table_lines:
        out.append(render_table(table_lines))

    return "\n".join(out)


def detect_paragraph_class(text):
    """简单判断段落是否主要为中文"""
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    if chinese_chars > len(text) * 0.3:
        return "zh"
    return "en"


def render_table(lines):
    if len(lines) < 2:
        return ""
    html = '<table style="width:100%;border-collapse:collapse;margin:15px 0">'
    for i, row in enumerate(lines):
        cells = [c.strip() for c in row.strip("|").split("|")]
        if i == 0:
            html += '<tr style="background:#3498db;color:white">'
            tag = "th"
        else:
            bg = "#f8f9fa" if i % 2 == 0 else "#fff"
            html += f'<tr style="background:{bg}">'
            tag = "td"
        for cell in cells:
            border = "border-bottom:1px solid #eee;padding:10px 12px;text-align:left"
            html += f'<{tag} style="{border}">{cell}</{tag}>'
        html += "</tr>"
    html += "</table>"
    return html


# ── 步骤6：发送邮件 ───────────────────────────────────

EMAIL_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif;
      max-width: 700px; margin: 0 auto; padding: 20px; background: #f5f6fa; }
.container { background: #fff; border-radius: 12px; padding: 30px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.header { text-align: center; margin-bottom: 30px; }
.header h1 { color: #2c3e50; font-size: 26px; margin-bottom: 5px; }
.header p { color: #95a5a6; font-size: 14px; }
h2 { color: #2c3e50; font-size: 20px; margin-top: 30px; }
h3 { color: #34495e; font-size: 17px; margin-top: 20px; }
p { line-height: 1.9; color: #333; font-size: 15px; }
p.en { color: #2c3e50; }
p.zh { color: #444; background: #eef2f7; padding: 12px 16px; border-radius: 6px;
       border-left: 4px solid #3498db; }
table { font-size: 14px; }
th { font-size: 14px; }
td { font-size: 14px; }
.footer { text-align: center; color: #b0b0b0; font-size: 12px; margin-top: 40px;
          padding-top: 20px; border-top: 1px solid #eee; }
"""


def build_email_html(translated_articles):
    date_str = datetime.now().strftime("%Y年%m月%d日")
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][datetime.now().weekday()]
    date_display = f"{date_str} {weekday}"

    body_html = ""
    for i, md in enumerate(translated_articles):
        if md is None:
            continue
        body_html += md_to_html(md)
        if i < len(translated_articles) - 1:
            body_html += '<hr style="border:none;border-top:2px dashed #e0e0e0;margin:40px 0">'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{EMAIL_CSS}</style></head>
<body>
<div class="container">
  <div class="header">
    <h1> Daily English Reader</h1>
    <p>{date_display} · 每日英语阅读 · 金融/科技/治理/政经</p>
  </div>
  {body_html}
  <div class="footer">
    <p>由 DeepSeek AI 自动生成 · 仅供个人学习用途</p>
    <p>文章版权归原作者所有</p>
  </div>
</div>
</body></html>"""


def send_email(html):
    recipients = [r.strip() for r in RECEIVER_EMAIL.split(",") if r.strip()]

    msg = MIMEMultipart("alternative")
    date_str = datetime.now().strftime("%Y-%m-%d")
    msg["Subject"] = f" Daily English Reader - {date_str}"
    msg["From"] = f"Daily English Reader <{SENDER_EMAIL}>"
    msg["To"] = ", ".join(recipients)

    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        print(f"  邮件发送成功! 收件人: {len(recipients)} 人")
    except Exception as e:
        print(f"  邮件发送失败: {e}")


# ── 主流程 ────────────────────────────────────────────

SEND_INTERVAL_DAYS = 3  # 每隔多少天发一封邮件


def should_send_today():
    """判断今天是否为发送日（从固定锚点起每 N 天一次）"""
    anchor = date(2025, 1, 1)  # 固定参考日期
    days_diff = (date.today() - anchor).days
    return days_diff % SEND_INTERVAL_DAYS == 0


def main():
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"  Daily English Reader - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if not should_send_today():
        print(f"[Skip] 今天是第 {(date.today() - date(2025, 1, 1)).days} 天，"
              f"不是发送日（每 {SEND_INTERVAL_DAYS} 天一次），退出")
        return

    # 1. RSS
    print("[1/5] 抓取 RSS ...")
    articles = fetch_articles()

    if not articles:
        print("  没有抓取到文章，退出")
        return

    # 2. AI 筛选
    print(f"\n[2/5] AI 筛选 (兴趣: {INTERESTS}) ...")
    selected = ai_filter(articles)
    print(f"  选中 {len(selected)} 篇:")
    for a in selected:
        print(f"    [{a['source']}] {a['title'][:60]}")
        print(f"     理由: {a.get('ai_reason', 'N/A')}")

    # 3. 爬正文
    print(f"\n[3/5] 爬取正文 ...")
    for a in selected:
        print(f"  {a['title'][:50]}...")
        body = fetch_body(a["link"])
        a["body"] = body
        print(f"    {len(body)} chars" if body else "    失败")
        time.sleep(2)

    selected = [a for a in selected if a.get("body")]

    if not selected:
        print("  所有文章正文抓取失败，退出")
        return

    # 4. AI 翻译
    print(f"\n[4/5] AI 翻译排版 ...")
    translated = []
    for a in selected:
        print(f"  翻译: {a['title'][:50]}...")
        result = ai_translate(a)
        if result:
            translated.append(result)
            print(f"    完成 ({len(result)} chars)")
        else:
            print(f"    失败")
        time.sleep(1)

    if not translated:
        print("  没有成功翻译的文章，退出")
        return

    # 5. 发送
    print(f"\n[5/5] 组装邮件并发送 ...")
    html = build_email_html(translated)
    send_email(html)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  全部完成! 耗时 {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
