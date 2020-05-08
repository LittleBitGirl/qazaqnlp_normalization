import re
import sys


class Classificator:
    errors_list = {}
    word = ''
    emoticons = ''

    def __init__(self, word):
        self.word = word
        with open('emoticons.txt', 'r') as file:
            emoticons = [line.strip() for line in file]
        self.emoticons = emoticons

    def isWord(self):
        return self.word.isalpha()

    def isEmoticon(self):
        isEmoticon = False
        for emoticon in self.emoticons:
            if emoticon == self.word:
                isEmoticon = True
        return isEmoticon

    def isHashOrMention(self):
        hash_pointer = self.word.find('#')
        ment_pointer = self.word.find('@')
        if hash_pointer != -1:
            self.errors_list['hashtag'] = hash_pointer
        elif ment_pointer != -1:
            self.errors_list['mention'] = ment_pointer
        return False

    def isVowelFrequency(self):
        match = re.search('([АаӘәІіҮүҰұӨөОоЕеИиЯяЫыЭэЮюИиУуЁё]{2,})', self.word)
        if match is not None:
            self.errors_list['vowelFrequency'] = match
            return True
        return False

    def isConsonantFrequency(self):
        match = re.search('([БбВвГгҒғДдЖжЗзКкҚқЛлМмНнҢңПпРрСсТтФфХхҺһЦцЧчШшЩщЪъЬь]{3,})', self.word)
        if match is not None:
            self.errors_list['consonantFrequency'] = match
            return True
        return False

    def classify(self, word):
        return word
