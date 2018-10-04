import konlpy
import pprint
import ujson

from konlpy.tag import Komoran
from collections import Counter

_NEWS_URL = 'http://news.naver.com/main/list.nhn?sid2=258&sid1=101&mid=shm&mode=LS2D'
_NEWS_URL_DIR = './URLs/News/Naver/'
_NEWS_TXT_DIR = './TXTs/News/Naver/'
_NEWS_CONT_DIR = './Contents/News/Naver/'
_NEWS_DATE = None


def split_text(text):

    new_text = text.replace(".",".\n").replace("?","?\n").replace("!","!\n")
    sentences = new_text.splitlines()

    return sentences

def get_morph_anal(analyzer, text):
    """
    형태소 분석을 하여 결과를 돌려 준다.
    """

    morph_anal = analyzer.pos(text, flatten=False)

    return morph_anal

def print_morph_anal(text, morph_anal, output_format):
    """
    형태소 분석 결과를 출력한다.
    """
    output_format = output_format.lower()
    if output_format =="json":
       output = get_json_output(text,morph_anal)
    elif output_format == "vert":
        output = get_vert_output(text, morph_anal)
    elif output_format == "hori":
        output = get_hori_output(text,morph_anal)

    print(output)

def get_json_output(text, morph_anal):
    """
    텍스트의 분석결과를 json 형식의 문자열로 만든다.
    """

    outputObj={
        "text" : text,
        "morphAnal" : morph_anal
    }
    output = ujson.dumps(outputObj,ensure_ascii=False)

    return output

def get_vert_output(text, morph_anal):
    """
    텍스트 분석 결과를 수직 형식의 문자열로 만든다.
    """

    vert_elems = []
    wordforms = text.split()

    for wordform, wordform_anal in zip(wordforms, morph_anal):
        morphs = []

        for lex, pos in wordform_anal:
            morphs.append(lex + "/" + pos)

        vert_elems.append(wordform + "\t" + "+".join(morphs))

    output = "\n".join(vert_elems)

    return output

def get_hori_output(text, morph_anal):
    """
    텍스트 분석 결과를 수평 형식의 문자열로 만든다.
    """

    hori_elems = []
    wordforms = text.split()

    for wordform, wordform_anal in zip(wordforms, morph_anal):
        morphs=[]

        for lex, pos in wordform_anal:
            morphs.append(lex + "/" + pos)

        hori_elems.append(wordform + "_" + "+".join(morphs))

    output = " ".join(hori_elems)

    return output

def main():
    """
    형태소 분석 결과를 텍스트 형식으로 저장한다.
    """
    file_name = _NEWS_CONT_DIR + '20170101.txt'
    komoran = Komoran()

    """
    with open(file_name,"r", encoding="utf-8") as f:
        for line in f:
            sentences = split_text(line)
            for sentence in sentences:
                morph_anal = get_morph_anal(komoran,sentence)
                print_morph_anal(sentence, morph_anal, "json")
                print_morph_anal(sentence, morph_anal, "vert")
                print_morph_anal(sentence, morph_anal, "hori")
    """

    with open(file_name, "r", encoding="utf-8") as f:
        noun_adj_list = []

        for line in f:
            sentences = split_text(line)
            for sentence in sentences:
                sentences_tag = komoran.pos(sentence)
                for words in sentences_tag:
                    word, tag = words
                    if tag in ['VCP', 'VCN']:
                        noun_adj_list.append(word)

        counts = Counter(noun_adj_list)
        print(counts.most_common(50))


main()