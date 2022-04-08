from tkinter import *
from tkinter.scrolledtext import ScrolledText
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from textblob import TextBlob
from spellchecker import SpellChecker
import random
from nltk.metrics.distance import edit_distance
import re
from collections import Counter
import string

#Corpus download and import
from nltk.corpus import words

# Corpus importer
def read_textfile(filename):
    with open(filename, "r", errors = 'ignore') as f:
        lines = f.readlines()
        words = []
        
        for line in lines: words = words + re.findall(r'\w+', line.lower())
        
    return words

textfile = read_textfile('MedicalExperience.txt')
textcorpuslist = [k.lower() for k in textfile]
uniq_vocab = set(textcorpuslist) # To list out words in the dictionary pane


#to calculate probability of each word based on total number of words in the corpus
word_count = Counter(textfile) 
total_word_freq = float(sum(word_count.values()))
word_probability = {word: word_count[word]/total_word_freq for word in word_count.keys()}


def ngram_dict(words): #method to convert a text into bigram. if trigram is used change the number after 'range' command
    # generates a list of Tuples representing all n-grams
    ngrams_tuple = zip(*[words[i:] for i in range(2)])
    # turn the list into a dictionary with the counts of all ngrams
    ngrams_count = {}
    for ngram in ngrams_tuple:
        if ngram not in ngrams_count:
            ngrams_count[ngram] = 0
        ngrams_count[ngram] += 1
    return ngrams_count

bigramdict = ngram_dict(textfile) #store all bigram words of the corpus into bigramdict

# all these to generate all possibilities of a word
def split(word):
    return [((word[:i], word[i:])) for i in range(len(word) + 1)]
def deletion(word):
    return [(l + r[1:]) for l,r in split(word) if r]
def insertion(word):
    letters = 'abcdefghiklmnopqrstuvwxyz'
    return [str(l + c + r) for l, r in split(word) for c in letters]
def subtitution(word):
    letters = 'abcdefghiklmnopqrstuvwxyz'
    return [str(l + c + r[1:]) for l, r in split(word) if r for c in letters]
def candidate1(word):
    return set(deletion(word) + subtitution(word) + insertion(word)) #few possible words will be extracted (apply for only small typo to the max ED of 2)
def candidate2(word):
    return set(e2 for e1 in candidate1(word) for e2 in candidate1(e1)) #more possible words will be extracted (apply for larger typo to the max ED of 4)

def candidates_list(word, vocab, word_probability): #Used in candidates function (output candidate box -right)
    possible_candidate = candidate1(word) or candidate2(word) or [word]
    best_words = [k for k in possible_candidate if k in vocab]
    
    def edit_distance(best_words):
        for index in best_words:
            target = index
            original = word
            target = [k for k in target]
            original = [k for k in original]
            row = len(target)
            column = len(original)
                
            matrix_table = np.zeros((row+1, column+1)) 
        
            matrix_table[0] = [j for j in range(column+1)]
            matrix_table[:,0] = [j for j in range(row+1)]
        
            if target[0] != original[0]:
                matrix_table[1,1] = 2 
                
            for c in range (1, column+1):
                    for r in range(1, row+1):
                        if target[r-1] != original[c-1]: 
                            matrix_table[r,c]= min(matrix_table[r-1,c], matrix_table[r,c-1]) + 1  
                        else:
                            matrix_table[r,c] = matrix_table[r-1,c-1]
            edit_distance = matrix_table[row,column]
            return(edit_distance)
        
    return [(k, edit_distance([k])) for k in best_words]


total_pair_word = float(sum(bigramdict.values()))
pair_probability = {pair : bigramdict[pair]/total_pair_word for pair in bigramdict.keys()}


def input_bigram(words): #Used in Correction function to convert input words into bigram
    # generates a list of Tuples representing all n-grams
    ngrams_tuple = zip(*[words[i:] for i in range(2)])
    # turn the list into a dictionary with the counts of all ngrams
    ngrams_count = {}
    for ngram in ngrams_tuple:
        if ngram not in ngrams_count:
            ngrams_count[ngram] = 0
        ngrams_count[ngram] += 1
    return ngrams_count

def correct_word_ngram(word, vocab, word_probability): #Used in Correction function to provide bigram guesses based on edit distance

    possible_candidate = candidate1(word) or candidate2(word) or [word]
    best_words = [k for k in possible_candidate if k in vocab]
    
    def edit_distance(best_words):
        for index in best_words:
            target = index
            original = word
            target = [k for k in target]
            original = [k for k in original]
            row = len(target)
            column = len(original)
                
            matrix_table = np.zeros((row+1, column+1)) 
        
            matrix_table[0] = [j for j in range(column+1)]
            matrix_table[:,0] = [j for j in range(row+1)]
        
            if target[0] != original[0]:
                matrix_table[1,1] = 2 
                
            for c in range (1, column+1):
                    for r in range(1, row+1):
                        if target[r-1] != original[c-1]: 
                            matrix_table[r,c]= min(matrix_table[r-1,c], matrix_table[r,c-1]) + 1  
                        else:
                            matrix_table[r,c] = matrix_table[r-1,c-1]
            edit_distance = matrix_table[row,column]
            return(edit_distance)
    return [(k) for k in best_words]

def edit_distance(best_words, original_word): #used in correction to generate edit distance
        target = best_words
        original = original_word
        target = [k for k in target]
        original = [k for k in original]
        row = len(target)
        column = len(original)
                
        matrix_table = np.zeros((row+1, column+1)) 
        
        matrix_table[0] = [j for j in range(column+1)]
        matrix_table[:,0] = [j for j in range(row+1)]
        
        if target[0] != original[0]:
            matrix_table[1,1] = 2 
                
        for c in range (1, column+1):
            for r in range(1, row+1):
                if target[r-1] != original[c-1]: 
                    matrix_table[r,c]= min(matrix_table[r-1,c], matrix_table[r,c-1]) + 1  
                else:
                    matrix_table[r,c] = matrix_table[r-1,c-1]
        edit_distance = matrix_table[row,column]
        return(edit_distance)


'______ Global initialization ______'
spell= SpellChecker()
window = Tk()
list = [w.lower() for w in textcorpuslist]
random.shuffle(list)
corp = list[:]

#GUI title
window.title('Spell Checker')
#GUI total dimension size
window.geometry("1000x750")



#Clear function
def clearfunction(): #Clear text inside the box
    Enter_text_box.delete('1.0',END)
    output_text.delete('1.0',END)
    output_Candidates.delete('1.0',END)
    candidate_text_box.delete('1.0',END)

def findreal(highlighted_word): #to highlight real word if found (blue)
    #Enter_text_box.tag_remove('found', '1.0' ,END)
    ser = highlighted_word
    if ser:
        idx = '1.0'
        while 1:
            idx = Enter_text_box.search(ser,idx,nocase=1, stopindex=END)
            if not idx:break
            lastidx = '%s+%dc' % (idx, len(ser))
            
            Enter_text_box.tag_add('foundreal', idx, lastidx)
            idx = lastidx
        Enter_text_box.tag_config('foundreal', foreground = 'blue')

def findnonreal(highlighted_word): #to highlight non real word if found (green)
    #Enter_text_box.tag_remove('found', '1.0' ,END)
    ser = highlighted_word
    if ser:
        idx = '1.0'
        while 1:
            idx = Enter_text_box.search(ser,idx,nocase=1, stopindex=END)
            if not idx:break
            lastidx = '%s+%dc' % (idx, len(ser))
            
            Enter_text_box.tag_add('foundnon', idx, lastidx)
            idx = lastidx
        Enter_text_box.tag_config('foundnon', foreground = 'green')


#Main Spell Correction Function
def correction1(e):
    output_text.delete('1.0',END)
    input_word = Enter_text_box.get("1.0", "end-1c")
    input_word_list = [k.lower() for k in input_word.split()]
    input_bigram_form = input_bigram(input_word_list)

    for scan in input_word_list:
        if scan not in uniq_vocab:
            findnonreal(scan)
            
    for first in input_bigram_form:
        if first not in pair_probability:
            correct = first[0]
            wrong = first[1]
            guess = correct_word_ngram(wrong,uniq_vocab,word_probability)
            #print(guess)
            wrong_real_word = []
            for pair in bigramdict.keys():
                for k in guess:
                    if pair[1] == k and pair[0] == correct:
                        #print(pair[1])
                        output_text.insert(END, f'{first[1]} : {pair[1]} : {edit_distance(pair[1],first[1])}\n')
                        findreal(first[1])
                        #output_MED.insert(END,f'{first[1]}\n')
                        
                        
                    elif pair[0] == k and pair[1] == correct:
                        #print(pair[0])
                        output_text.insert(END, f'{first[1]} : {pair[0]} : {edit_distance(pair[0],first[1])}\n')
                        findreal(first[1])
                        #output_MED.insert(END,f'{first[1]}\n')

def candidatescheck():
    output_Candidates.delete('1.0',END)
    input_word = candidate_text_box.get("1.0", "end-1c")
    input_word_list = [k.lower() for k in input_word.split()]
    

    output_Candidates.insert(END, '\n'.join(map(str,candidates_list(input_word_list[0],uniq_vocab,word_probability))))
    



#INPUT TEXT(LEFT) widget (line 106 - 119)
Input_Text_label = Label(window,text = 'Input text (Max 500 Words)', font = ('Tmes New Roman',))
Input_Text_label.place(x = 2, y = 0)

Enter_text_box = Text(window, fg = 'black', bg = 'silver', width  = 60, height = 16, foreground = 'red', font = ('Times New Roman', 15))
Enter_text_box.place(x = 2,y = 25)


clear_button = Button(window, text='Clear', padx = 30, pady = 10,background = 'orange',foreground = 'white', command = clearfunction)
clear_button.place(x = 650, y = 240)

#Enter key function
window.bind('<Return>', correction1)

#Dictionary(RIGHT) widget (line 122 - 152)
def Scankey(event):
	val = event.widget.get()
	print(val)
	if val == '':
		data = uniq_vocab
	else:
		data = []
		for item in uniq_vocab:
			if val.lower() in item.lower():
				data.append(item)					
	Update(data)

def Update(data):	
	listbox.delete(0, 'end')
	# put new data
	for item in data:
		listbox.insert('end', item)

rightframe = Frame(window)
rightframe.pack(side = RIGHT, fill = BOTH)

Input_Text_label = Label(rightframe,text = 'Dictionary', font = ('Tmes New Roman',))
Input_Text_label.pack(side = TOP)

Enter_text_box_search = Entry(rightframe, fg = 'black', bg = 'silver', width  = 20, foreground = 'red', font = ('Times New Roman',))
Enter_text_box_search.pack(side = TOP)
Enter_text_box_search.bind('<KeyRelease>', Scankey)

listbox  = Listbox(rightframe, fg = 'black', bg = 'silver', width = 20, height  = 17, foreground = 'red', font = ('Times New Roman',))
listbox.pack(side = TOP)
Update(uniq_vocab)

#OUTPUT widget  (line 155 - 158)
output_label = Label(window,text = 'Output', font = ('Tmes New Roman',))
output_label.place(x= 2, y = 400)
output_text = ScrolledText(window, fg = 'black', bg = 'silver', width  = 60, height = 14, foreground = 'red', font = ('Times New Roman', 15))
output_text.place(x = 2,y = 425)

#MINIMUM EDIT DISTANCE OUTPUT widget (line 161 - 164)
output_label_candidates = Label(window,text = 'Possible Candidates', font = ('Tmes New Roman',))
output_label_candidates.place(x= 800, y = 400)
output_Candidates = ScrolledText(window, fg = 'black', bg = 'silver', width  = 18, height = 14, foreground = 'red', font = ('Times New Roman', 15))
output_Candidates.place(x=800, y= 425)
candidate_text_box = Text(window,fg = 'black', bg = 'silver', width  = 10,height = 1, foreground = 'red', font = ('Times New Roman', 15))
candidate_text_box.place(x = 650, y= 430)
candidate_button = Button(window, text = 'Candidate',padx = 19, pady=10,foreground= 'white',background = 'green',command = candidatescheck)
candidate_button.place( x= 650, y = 465)

window.mainloop()
