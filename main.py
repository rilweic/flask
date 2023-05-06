# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：2023/3/16 11:15
# Author     ：lichao
"""
import base64
import hashlib
import hmac
import json
import threading
import time

import asyncio
import openai
import os

import requests
from EdgeGPT import Chatbot, ConversationStyle
from bs4 import BeautifulSoup
from flask import request, Flask, jsonify
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from queue import Queue
from threading import Thread, Lock
import pytz
from datetime import datetime
from functools import wraps

app = Flask(__name__)
task_queue = Queue()

EXPIRE_TIME = int(time.time()) - 1
BING_COUNT = 0

lock_doreamon = Lock()
lock_echo = Lock()
lock_bing = Lock()
lock_picasso = Lock()

conf = {
    "dingding_echo_app_secret": os.environ.get('dingding_echo_app_secret'),
    "dingding_doraemon_app_secret": os.environ.get('dingding_doraemon_app_secret'),
    "dingding_picasso_app_secret": os.environ.get('dingding_picasso_app_secret'),
    "dingding_bing_app_secret": os.environ.get('dingding_bing_app_secret'),

    "rgzn_dingding_echo_app_secret": os.environ.get('rgzn_dingding_echo_app_secret'),
    "rgzn_dingding_doraemon_app_secret": os.environ.get('rgzn_dingding_doraemon_app_secret'),
    "rgzn_dingding_picasso_app_secret": os.environ.get('rgzn_dingding_picasso_app_secret'),
    "rgzn_dingding_bing_app_secret": os.environ.get('rgzn_dingding_bing_app_secret'),

    "poe_form_key": os.environ.get('poe_form_key'),
    "poe_cookie": os.environ.get('poe_cookie'),
    "chatgpt_apy_key": os.environ.get('chatgpt_apy_key'),
}

cookie_file_path = os.path.join(app.root_path, 'cookies.json')

CHATGPT_APY_KEY = conf['chatgpt_apy_key']


# 钉钉工具类
class DingdingUtil(object):

    @classmethod
    def check_sig(cls, timestamp, app_secret):
        app_secret_enc = app_secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, app_secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(app_secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = base64.b64encode(hmac_code).decode('utf-8')
        return sign

    @classmethod
    def sendMarkdown(cls, userid, title, message, webhook_url):
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": message
            },
            "at": {
                "atUserIds": [
                    userid
                ]
            }
        }
        # 利用requests发送post请求
        resp = requests.post(webhook_url, json=data)
        if resp.status_code == 200:
            print("钉钉消息发送成功")
        else:
            print(resp.status_code)


# openapi官方api
class OpenAIFunction(object):
    def __init__(self, gpt_model="gpt-3.5-turbo", api_key=None,
                 img_dir=None):
        self.gpt_model = gpt_model
        self.api_key = api_key
        if img_dir is None:
            img_dir = "images"
        self.img_dir = os.path.join(os.curdir, img_dir)

    def ask(self, question) -> str:
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Knock knock."},
                {"role": "assistant", "content": "Who's there?"},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        return response['choices'][0]['message']['content']

    # 生成一张图片，返回图片的url地址
    def generate_pic(self, prompt):
        openai.api_key = self.api_key
        resp = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url",
        )
        return resp['data'][0]['url']

    def request_chatgpt_server(self, msg):
        data = {
            "message": msg,
        }
        url = "http://localhost:{}/message/holegots".format(4000)
        # 利用requests发送post请求
        headers = {'Content-Type': 'application/json; charset=UTF-8'}
        rep = requests.post(url, headers=headers, json=data)
        print(rep.status_code)
        if rep.status_code == 200:
            response = rep.json()["response"]
        else:
            response = "出错了，状态码为{},错误消息为 {}".format(rep.status_code, rep.text)

        return response


class CipherAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ciphers='DEFAULT:@SECLEVEL=2')
        kwargs['ssl_context'] = context
        return super(CipherAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = create_urllib3_context(ciphers='DEFAULT:@SECLEVEL=2')
        kwargs['ssl_context'] = context
        return super(CipherAdapter, self).proxy_manager_for(*args, **kwargs)


##########################################################################

# 免费的stable diffusion图片生成
class FreeStableDuffision(object):

    def __init__(self, version='328bd9692d29d6781034e3acab8cf3fcb122161e6f5afb896a4ca9fd57090577'):
        self.version = version

    # stable diffusion 生成一张图片，返回图片的url地址
    def generate_pic(self, propmt):
        data = {
            "inputs":
                {
                    "prompt": propmt,
                    "scheduler": "K_EULER", "num_outputs": "1",
                    "guidance_scale": 7.5,
                    "image_dimensions": "768x768",
                    "num_inference_steps": 50
                }
        }
        req_client = requests.Session()
        req_client.mount('https://replicate.com', CipherAdapter())
        req_client.headers[
            "User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1660.13"
        url = "https://replicate.com/api/models/stability-ai/stable-diffusion/versions/{}/predictions".format(
            self.version)
        resp = req_client.post(url=url, json=data)
        if resp.status_code >= 200 and resp.status_code < 300:
            p_uuid = resp.json()["uuid"]
        else:
            req_client.close()
            return ''
        q_url = "https://replicate.com/api/models/stability-ai/stable-diffusion/versions/{}/predictions/{}".format(
            self.version, p_uuid)

        while True:
            tmp_resp = req_client.get(q_url)
            if tmp_resp.status_code >= 300:
                break
            else:
                if tmp_resp.json()['prediction']['status'] != "succeeded":
                    time.sleep(2)
                else:
                    output = tmp_resp.json()['prediction']['output']
                    if len(output) > 0:
                        req_client.close()
                        return output[0]


##########################################################################

# phind 搜索结果
class PhindSearchResult(object):
    def __init__(self, content, web_pages, suggestions):
        self.content = content
        self.web_pages = web_pages
        self.suggestions = suggestions

    def __repr__(self):
        return f"SearchResult(content={self.content}, web_pages={self.web_pages}, suggestions={self.suggestions})"


class PhindSearch(object):
    def __init__(self):
        self.base_url = 'https://phind.com'
        self.phind_search = "https://phind.com/api/bing/search"
        self.phind_tldr = "https://phind.com/api/infer/detailed"
        self.phind_suggested = "https://phind.com/api/infer/followup/suggestions"

    def search_phind(self, key_word):
        data = {"freshness": "",
                "q": key_word,
                "userRankList":
                    {
                        "developer.mozilla.org": 1,
                        "github.com": 1,
                        "stackoverflow.com": 1,
                        "www.reddit.com": 1,
                        "en.wikipedia.org": 1,
                        "www.amazon.com": -1,
                        "www.quora.com": -2,
                        "www.pinterest.com": -3,
                        "rust-lang": 2,
                        "google.com": 3,
                        ".rs": 1
                    }
                }

        client = requests.Session()
        client.mount(self.base_url, CipherAdapter())
        client.headers["User-Agent"] = "Mozilla/5.0"

        r = client.post(self.phind_search, json=data, stream=True)
        print("first phind search code:{}".format(r.status_code))
        if r.status_code == 200 and r.json():
            tldr_data = {
                "question": key_word,
                "bingResults": r.json()['rawBingResults']
            }
            resp = client.post(self.phind_tldr, json=tldr_data, stream=True)
            print("second phind search code:{}".format(r.status_code))
            if resp.status_code == 200:
                result = self.process_resp(resp.text)
            else:
                ai_client = OpenAIFunction(api_key=CHATGPT_APY_KEY)
                result = ai_client.ask(key_word)
            # 查询 搜索建议
            suggest_data = {"question": key_word, "answer": result}
            suggest_resp = client.post(self.phind_suggested, json=suggest_data, stream=True)
            web_pages = r.json()['processedBingResults']['webPages']['value'][:10]
            suggest_list = json.loads(suggest_resp.text)
            client.close()
            phind_search_result = PhindSearchResult(result, web_pages, suggest_list)
            return phind_search_result
        else:
            client.close()
            return None

    def process_resp(self, text):
        ll = text.split("\r\n\r\n")
        all_list = []
        for i in ll:
            if i:
                try:
                    data = i[6:]
                    all_list.append(data)
                except Exception as e:
                    continue
        return "".join(all_list)


##########################################################################

class SearchResult:
    def __init__(self, url, title, description):
        self.url = url
        self.title = title
        self.description = description

    def __repr__(self):
        return f"SearchResult(url={self.url}, title={self.title}, description={self.description})"


class SearchGoogle(object):
    usr_agent = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
    url = "https://www.google.com/search?gl=us|cn&lr=lang_en|lang_zh-CN|lang_zh-TW"

    def _req(self, term, results, lang, start, proxies):
        resp = requests.get(
            url=self.url,
            headers=self.usr_agent,
            params=dict(
                q=term,
                num=results + 2,  # Prevents multiple requests
                hl=lang,
                start=start,
            ),
            proxies=proxies,
        )
        return resp

    def check_429_expire(self):
        global EXPIRE_TIME
        return int(time.time()) > EXPIRE_TIME

    # 返回搜索列表
    def search(self, term, num_results=10, lang="en", proxy=None, advanced=False) -> list:
        escaped_term = term.replace(' ', '+')

        # Proxy
        proxies = None
        if proxy:
            if proxy[:5] == "https":
                proxies = {"https": proxy}
            else:
                proxies = {"http": proxy}
        search_result = []
        # Fetch
        start = 0
        if not self.check_429_expire():
            print("请求过于频繁，请稍后重试")
            return search_result
        while start < num_results:
            # Send request
            resp = self._req(escaped_term, num_results - start, lang, start, proxies)
            if resp.status_code == 200:
                # Parse
                soup = BeautifulSoup(resp.text, 'html.parser')
                result_block = soup.find_all('div', attrs={'class': 'g'})
                for result in result_block:
                    # Find link, title, description
                    link = result.find('a', href=True)
                    title = result.find('h3')
                    description_box = result.find('div', {'style': '-webkit-line-clamp:2'})
                    if description_box:
                        description = description_box.find('span')
                        if link and title and description:
                            start += 1
                            if advanced:
                                search_result.append(SearchResult(link['href'], title.text, description.text))
            elif resp.status_code == 429:
                global EXPIRE_TIME
                EXPIRE_TIME = int(time.time()) + 3600
                break
            else:
                print("系统错误")
                break

        return search_result


##########################################################################
class Echo(object):
    form_key = conf['poe_form_key']
    cookie = conf['poe_cookie']

    def __init__(self, default_bot=2):
        self.url = 'https://www.quora.com/poe_api/gql_POST'
        self.headers = {
            'Host': 'www.quora.com',
            'Accept': '*/*',
            'apollographql-client-version': '1.1.6-65',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'User-Agent': 'Poe 1.1.6 rv:65 env:prod (iPhone14,2; iOS 16.2; zh-CN)',
            'apollographql-client-name': 'com.quora.app.Experts-apollo-ios',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Quora-Formkey': self.form_key,
            'Cookie': self.cookie
        }
        self.default_bot = default_bot

        # print("1. Claude - Anthropic (a2)")
        # print("2. ChatGPT-Big - OpenAI (capybara)")
        # print("3. ChatGPT-Small - Openai (nutria)")
        # print("4. ChatGPT-Big - Openai (chinchilla)")
        self.bot = None
        self.chat_id = None

        self.init_bot(self.default_bot)

    def set_auth(self, key, value):
        self.headers[key] = value

    def load_chat_id_map(self, bot="a2"):
        data = {
            'operationName': 'ChatViewQuery',
            'query': 'query ChatViewQuery($bot: String!) {\n  chatOfBot(bot: $bot) {\n    __typename\n    ...ChatFragment\n  }\n}\nfragment ChatFragment on Chat {\n  __typename\n  id\n  chatId\n  defaultBotNickname\n  shouldShowDisclaimer\n}',
            'variables': {
                'bot': bot
            }
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        c_id = response.json()['data']['chatOfBot']['chatId']

        return c_id

    def send_message(self, message, bot="a2", chat_id=""):
        data = {
            "operationName": "AddHumanMessageMutation",
            "query": "mutation AddHumanMessageMutation($chatId: BigInt!, $bot: String!, $query: String!, $source: MessageSource, $withChatBreak: Boolean! = false) {\n  messageCreate(\n    chatId: $chatId\n    bot: $bot\n    query: $query\n    source: $source\n    withChatBreak: $withChatBreak\n  ) {\n    __typename\n    message {\n      __typename\n      ...MessageFragment\n      chat {\n        __typename\n        id\n        shouldShowDisclaimer\n      }\n    }\n    chatBreak {\n      __typename\n      ...MessageFragment\n    }\n  }\n}\nfragment MessageFragment on Message {\n  id\n  __typename\n  messageId\n  text\n  linkifiedText\n  authorNickname\n  state\n  vote\n  voteReason\n  creationTime\n  suggestedReplies\n}",
            "variables": {
                "bot": bot,
                "chatId": chat_id,
                "query": message,
                "source": None,
                "withChatBreak": False
            }
        }
        _ = requests.post(self.url, headers=self.headers, json=data)

    def clear_context(self, chatid):
        data = {
            "operationName": "AddMessageBreakMutation",
            "query": "mutation AddMessageBreakMutation($chatId: BigInt!) {\n  messageBreakCreate(chatId: $chatId) {\n    __typename\n    message {\n      __typename\n      ...MessageFragment\n    }\n  }\n}\nfragment MessageFragment on Message {\n  id\n  __typename\n  messageId\n  text\n  linkifiedText\n  authorNickname\n  state\n  vote\n  voteReason\n  creationTime\n  suggestedReplies\n}",
            "variables": {
                "chatId": chatid
            }
        }
        _ = requests.post(self.url, headers=self.headers, json=data)

    def get_latest_message(self, bot):
        data = {
            "operationName": "ChatPaginationQuery",
            "query": "query ChatPaginationQuery($bot: String!, $before: String, $last: Int! = 10) {\n  chatOfBot(bot: $bot) {\n    id\n    __typename\n    messagesConnection(before: $before, last: $last) {\n      __typename\n      pageInfo {\n        __typename\n        hasPreviousPage\n      }\n      edges {\n        __typename\n        node {\n          __typename\n          ...MessageFragment\n        }\n      }\n    }\n  }\n}\nfragment MessageFragment on Message {\n  id\n  __typename\n  messageId\n  text\n  linkifiedText\n  authorNickname\n  state\n  vote\n  voteReason\n  creationTime\n}",
            "variables": {
                "before": None,
                "bot": bot,
                "last": 1
            }
        }
        while True:
            time.sleep(2)
            response = requests.post(self.url, headers=self.headers, json=data)
            response_json = response.json()
            text = response_json['data']['chatOfBot']['messagesConnection']['edges'][-1]['node']['text']
            state = response_json['data']['chatOfBot']['messagesConnection']['edges'][-1]['node']['state']
            author_nickname = response_json['data']['chatOfBot']['messagesConnection']['edges'][-1]['node'][
                'authorNickname']
            if author_nickname == bot and state == 'complete':
                break
        return text

    def init_bot(self, index):
        bots = {1: 'a2', 2: 'capybara', 3: 'nutria', 4: 'chinchilla'}
        self.bot = bots[index]
        self.chat_id = self.load_chat_id_map(self.bot)

    def ask_echo(self, msg):
        self.send_message(msg, self.bot, self.chat_id)
        echo_resp = self.get_latest_message(self.bot)
        return echo_resp


class MsgWrapper(object):

    @classmethod
    def wrap_markdown_msg(cls, question, msg):
        if len(msg) == 0:
            return ''
        title = """{} 回复：{} """.format('AI', question if len(question) <= 12 else "{}...".format(question[:12]))
        return """ > {} \n __________________________ \n {}""".format(title, msg)

    @classmethod
    def wrap_bing_links(cls, bing_web_pages):
        res = ''
        for d in bing_web_pages:
            res = res + "* [{}]({}) {}".format(d['name'], d['url'], '\r\n')
        return res

    @classmethod
    def wrap_list_item(cls, list_items):
        ans = ''
        for s in list_items:
            ans = ans + "* {} {}".format(s, "\n")
        return ans

    @classmethod
    def wrap_google_links(cls, google_web_pages):
        res = ''
        for d in google_web_pages:
            res = res + "* [{}]({}) {}".format(d.title, d.url, '\r\n')
        return res

    @classmethod
    def wrap_link_title(cls, title, msg):
        return """ ### {} \n __________________________ \n{}\n \r\n """.format(title, msg)

    @classmethod
    def wrap_markdown_pic(cls, title, url):
        tt = """{} """.format(title if len(title) <= 12 else "{}...\n __________________________ \n".format(
            title[:12]))
        return "{}![{}]({})".format(tt, title, url)

    @classmethod
    def process_new_bing_response(cls, resp_dict):
        print(json.dumps(resp_dict))
        first_new_message_index = resp_dict['item']['firstNewMessageIndex']
        msgs = resp_dict['item']['messages']
        if len(msgs) == 0:
            return "答不上来，换个问题或者问问别的机器人吧，群里有个Echo，它一直活着，老铁"
        resp_msg = msgs[first_new_message_index]
        text = resp_msg['text']
        adaptive_cards = resp_msg['adaptiveCards']
        if len(adaptive_cards) == 0:
            if len(text) > 0:
                return text
            return "答不上来，换个问题或者问问别的机器人吧，群里有个Echo，它一直活着，老铁"
        body = adaptive_cards[0]['body']
        all_msg = ''
        for text_block in body:
            all_msg = all_msg + text_block['text'] + "\n"
        return all_msg


print("#################################### 初始化各个机器人 ######################################")

ai_cli = OpenAIFunction(api_key=CHATGPT_APY_KEY)
phind = PhindSearch()
echo = Echo()
sd = FreeStableDuffision()
g_search = SearchGoogle()
bing = await Chatbot.create(cookie_path=cookie_file_path)
# bing = Chatbot(cookiePath=cookie_file_path)
print("#################################### 完成各个机器人初始化 ######################################")


def log_args_and_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取当前时间

        # 将当前时间转换为东八区时间
        tz_eastern = pytz.timezone('Asia/Shanghai')
        time_eastern = datetime.now(tz_eastern)

        print("{} {} 正在处理 【{}】 发送的消息 {} ................".format(time_eastern.strftime('%Y-%m-%d %H:%M:%S'), func.__name__, args[0]['senderNick'],
                                                                   args[0]['text']['content'].strip()[:100]))
        result = func(*args, **kwargs)
        return result

    return wrapper


@log_args_and_time_decorator
def process_doraemon(req_data):
    text_info = req_data['text']['content'].strip()
    webhook_url = req_data['sessionWebhook']
    senderid = req_data['senderId']
    is_chatgpt = False

    if str(text_info[:7]).lower() == "chatgpt":
        text_info = str(text_info[7:]).strip()
        is_chatgpt = True
    tt = """{} 回复：{} """.format('AI', text_info if len(text_info) <= 12 else "{}...".format(text_info[:12]))
    if is_chatgpt:
        # ans = ai_cli.ask(text_info)
        poe_bot = Echo(4)  # 4默认为ChatGPT
        ans = poe_bot.ask_echo(text_info)
        # ans = ai_cli.request_chatgpt_server(text_info)
        # 发送ChatGPT的消息
        DingdingUtil.sendMarkdown(senderid, tt, MsgWrapper.wrap_markdown_msg(text_info, ans), webhook_url)
        poe_bot.clear_context(poe_bot.chat_id)
    else:
        phind_result = None
        bing_pages = ''
        suggest_items = ''
        content = ''

        try:
            phind_result = phind.search_phind(text_info)
        except Exception as phind_error:
            print(phind_error)
        if phind_result is not None:
            bing_pages = MsgWrapper.wrap_bing_links(phind_result.web_pages)
            suggest_items = MsgWrapper.wrap_list_item(phind_result.suggestions)
            content = MsgWrapper.wrap_markdown_msg(text_info, phind_result.content)
        google_pages = g_search.search(text_info, 15, advanced=True)
        if len(google_pages) > 0:
            google_pages = MsgWrapper.wrap_google_links(google_pages)

        if len(content) > 0:
            # 先发具体内容
            DingdingUtil.sendMarkdown(senderid, tt, content, webhook_url)

        more_info = ""
        if len(google_pages) > 0:
            more_info = more_info + MsgWrapper.wrap_link_title("谷歌搜索：", google_pages)
        if len(bing_pages):
            more_info = more_info + MsgWrapper.wrap_link_title("必应搜索：", bing_pages)
        if len(suggest_items):
            more_info = more_info + MsgWrapper.wrap_link_title("猜你想知道：", suggest_items)
        # 再发更多链接
        DingdingUtil.sendMarkdown(senderid, tt, more_info, webhook_url)


@log_args_and_time_decorator
def process_echo(req_data):
    text_info = req_data['text']['content'].strip()
    webhook_url = req_data['sessionWebhook']
    senderid = req_data['senderId']

    tt = """{} 回复：{} """.format('AI', text_info if len(text_info) <= 12 else "{}...".format(text_info[:12]))
    ans = echo.ask_echo(text_info)
    # 发送echo的钉钉消息
    DingdingUtil.sendMarkdown(senderid, tt, MsgWrapper.wrap_markdown_msg(text_info, ans), webhook_url)


@log_args_and_time_decorator
def process_picasso(req_data):
    text_info = req_data['text']['content'].strip()
    webhook_url = req_data['sessionWebhook']
    senderid = req_data['senderId']

    tt = """{} 回复：{} """.format('AI', text_info if len(text_info) <= 12 else "{}...".format(text_info[:12]))
    try:
        en_propmt = ai_cli.ask("翻译成英文：{}".format(text_info))
        print("翻译结果:" + en_propmt)
    except Exception as translate_error:
        print("translate_error:{}".format(translate_error))
        en_propmt = text_info
    try:
        pic_url = sd.generate_pic(en_propmt)
    except Exception as dalle:
        print('dalle:{}'.format(dalle))
        pic_url = ai_cli.generate_pic(en_propmt)

    DingdingUtil.sendMarkdown(senderid, tt, MsgWrapper.wrap_markdown_pic(tt, pic_url), webhook_url)


@log_args_and_time_decorator
async def process_bing(req_data):
    text_info = req_data['text']['content'].strip()
    webhook_url = req_data['sessionWebhook']
    senderid = req_data['senderId']

    tt = """{} 回复：{} """.format('AI', text_info if len(text_info) <= 12 else "{}...".format(text_info[:12]))
    # 设置重置次数
    global BING_COUNT
    if BING_COUNT > 10:
        await bing.reset()
    BING_COUNT = BING_COUNT + 1
    resp_dict = await bing.ask(text_info, conversation_style=ConversationStyle.creative)
    try:
        result = MsgWrapper.process_new_bing_response(resp_dict)
    except Exception as e:
        print(e)
        result = "😭 答不上来，换个问题或者问问别的机器人吧，群里有个Echo，它一直活着，老铁"
    DingdingUtil.sendMarkdown(senderid, tt, result, webhook_url)

@app.route("/", methods=["GET"])
def index():
    return "AAAAAAAAAAAAAAAIIIIIIIIIIIIIII"


@app.route("/<robot>", methods=["POST"])
def processer(robot):
    if request.method == "POST":
        app_key = None
        # 根据不同的机器人获取不同的apikey
        if robot == "doraemon":
            app_key = conf['dingding_doraemon_app_secret']
        elif robot == 'echo':
            app_key = conf['dingding_echo_app_secret']
        elif robot == 'picasso':
            app_key = conf['dingding_picasso_app_secret']
        elif robot == 'bing':
            app_key = conf['dingding_bing_app_secret']
        elif robot == 'rgzndoraemon':
            app_key = conf['rgzn_dingding_doraemon_app_secret']
        elif robot == 'rgznecho':
            app_key = conf['rgzn_dingding_echo_app_secret']
        elif robot == 'rgznpicasso':
            app_key = conf['rgzn_dingding_picasso_app_secret']
        elif robot == 'rgznbing':
            app_key = conf['rgzn_dingding_bing_app_secret']

        timestamp = request.headers.get('Timestamp')
        sign = request.headers.get('Sign')
        # 第二步验证：签名是否有效
        if DingdingUtil.check_sig(timestamp, app_key) == sign:
            # 获取、处理数据
            req_data = json.loads(str(request.data, 'utf-8'))
            # 将数据加入队列
            task_queue.put({"robot": robot, "data": req_data})

            return jsonify({'status': 'Task added to queue'})

        print('验证不通过')
        return 'ppp'


def process_data(wrapper_request):
    if "doraemon" in wrapper_request['robot']:
        with lock_doreamon:
            process_doraemon(wrapper_request['data'])
    if "echo" in wrapper_request['robot']:
        with lock_echo:
            process_echo(wrapper_request['data'])
    if "picasso" in wrapper_request['robot']:
        with lock_picasso:
            process_picasso(wrapper_request['data'])
    if "bing" in wrapper_request['robot']:
        with lock_bing:
            asyncio.run(process_bing(wrapper_request['data']))
    task_queue.task_done()


# 从队列中获取数据并给到对应的处理函数处理
def process_queue():
    while True:
        # 加上try，防止线程因为异常退出
        try:
            if not task_queue.empty():
                wrapper_request = task_queue.get()
                task_thread = threading.Thread(target=process_data, args=(wrapper_request,))
                task_thread.start()
            else:
                time.sleep(1)
        except Exception as e:
            print(e)


print("#################################### 启动处理线程 ######################################")
thread = Thread(target=process_queue)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
