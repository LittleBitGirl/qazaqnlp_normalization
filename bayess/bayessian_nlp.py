import re
from tkinter import *

from collections import Counter
import codecs
import string, sys, array


def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(
    words(
            codecs.open('abay_joli_1_full.txt', 'r', 'utf_8').read().translate(str.maketrans('', '', string.punctuation))
            + codecs.open('aqbilek_full.txt',  'r', 'utf_8').read().translate(str.maketrans('', '', string.punctuation))
            + codecs.open('bir_ata_bala_full.txt',  'r', 'utf_8').read().translate(str.maketrans('', '', string.punctuation))
          )
    )
def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'аәбвгғдеёжзийкқлмнңоөпрстуүұфхһцчшщыіэюя'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def getAccuracy(word, word1):
    correct = 0
    len_for = len(word) < len(word1) and len(word) or len(word1)
    for x in range(len_for):
        if word[x][-1] == word1[x]:
            correct += 1
    return (1.0 * correct / len(word)) * 100.0


def retrieve_text():
    word = app_entry.get()
    corrected_word = correction(app_entry.get())
    app_entry.delete(0, END)
    app_entry.insert(0, corrected_word)
    print(corrected_word)
    print('Accuracy is about {}'.format(round(getAccuracy(word, corrected_word))))



if __name__ == "__main__":
    app_win = Tk()
    app_win.title("spell")
    app_label = Label(app_win, text="Қазақша сөзді жазыныз✍️")
    app_label.pack()
    app_entry = Entry(app_win)
    app_entry.pack()
    app_button = Button(app_win, text="Тексеру✔️", command=retrieve_text)
    app_button.pack()
    # Initialize GUI loop
    app_win.mainloop()