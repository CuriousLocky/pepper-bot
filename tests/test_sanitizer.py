from pepperbot.conversation.sanitizer import ResponseSanitizer


def test_strips_message_prefix_and_bot_name():
    sanitizer = ResponseSanitizer(["BlackPepper", "黑胡椒"])
    assert sanitizer.clean("[msg 8] BlackPepper (reply to msg 7): hello") == "hello"


def test_strips_chinese_colon_bot_prefix():
    sanitizer = ResponseSanitizer(["BlackPepper", "黑胡椒"])
    assert sanitizer.clean("黑胡椒：才不是担心你呢") == "才不是担心你呢"


def test_strips_xml_wrapper():
    sanitizer = ResponseSanitizer(["BlackPepper"])
    assert sanitizer.clean('<message id="m1">hi</message>') == "hi"


def test_extracts_telegram_reply_xml():
    sanitizer = ResponseSanitizer(["BlackPepper"])
    parsed = sanitizer.parse("<telegram_reply>hi &amp; bye</telegram_reply>")
    assert parsed.text == "hi & bye"
    assert parsed.retry is False


def test_extracts_last_assistant_history_message_xml():
    sanitizer = ResponseSanitizer(["BlackPepper"])
    parsed = sanitizer.parse(
        '<thread><message id="m1" role="user">question</message><message id="m2" role="assistant">answer</message></thread>'
    )
    assert parsed.text == "answer"
    assert parsed.retry is False


def test_invalid_xml_like_output_requests_retry():
    sanitizer = ResponseSanitizer(["BlackPepper"])
    parsed = sanitizer.parse("<telegram_reply>unfinished")
    assert parsed.text == ""
    assert parsed.retry is True


def test_does_not_strip_normal_sentence():
    sanitizer = ResponseSanitizer(["BlackPepper"])
    assert sanitizer.clean("BlackPepper is here: hello") == "BlackPepper is here: hello"
