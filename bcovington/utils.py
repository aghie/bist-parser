from collections import Counter
import re

"""
This is a module slightly extended from original utils in BIST-Parser:
https://github.com/elikip/bist-parser/blob/master/barchybrid/src/utils.py

that has been adapted to include to support non-projective transition-based dependency parsing
and CoNLLU dependencies.
"""

class CovingtonConfiguration(object):
    """
    Nivre, J. (2008). Algorithms for deterministic incremental dependency parsing. Computational Linguistics, 34(4), 513-553.
    
    l1: Word Id of the word at the top of the lambda one list
    b: Word Id of the word at the top of the buffer
    sentence: List of ConllEntry
    A: set of created arcs (tuples (headID,dependentID))
    """
    
    def __init__(self,l1,b,sentence, A):
        
        self.l1 = l1
        self.b = b
        self.sentence = sentence
        self.A = A
    
    def __str__(self):
        return str(self.l1)+" "+str(self.b)+" "+str(self.A)


class ConllEntry(object):
    """
    Contains the information of a line in a CoNLL-X file.
    """
    
    def __init__(self, id, form, lemma, cpos, pos, feats, 
                 parent_id=None, relation=None):
        
        self.id = id
        self.form = form
        self.lemma = normalize(lemma)
        self.norm = normalize(form)
        self.cpos = cpos
        self.pos = pos
        self.feats = feats
        self.parent_id = parent_id
        self.relation = relation

        #By default everything is assigned to a dummy root
        self.pred_parent_id = 0
        self.pred_relation = 'root'
    
    #For debugging
    def __str__(self):
        return "["+'\,'.join(map(str,[self.id,self.form,self.lemma,self.norm,self.cpos,self.pos,self.feats,self.parent_id,self.relation]))+"]"



def vocab(conll_path):
    
    wordsCount = Counter()
    lemmasCount = Counter()
    cposCount = Counter()
    posCount = Counter()
    featsCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):

            wordsCount.update([node.norm for node in sentence])
            lemmasCount.update([node.lemma for node in sentence])
            cposCount.update([node.cpos for node in sentence])
            posCount.update([node.pos for node in sentence])
            featsCount.update([node.feats for node in sentence])
            relCount.update([node.relation for node in sentence])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, 
            lemmasCount, {l: i for i, l in enumerate(lemmasCount.keys())},
            cposCount.keys(), posCount.keys(), featsCount.keys(), 
            relCount.keys())


def read_conll(fh):
    """
    Reads a ConLL file given a file object fh
    """
    
    non_proj_sentences = 0
    read = 0
    tokens_read = 0
    root = ConllEntry(0, '*root*', '*root-lemma*', 'ROOT-POS', 'ROOT-CPOS','FEATS-ROOT', 0, 'rroot')
    tokens = [root]
    for line in fh:
        
        if line.startswith('#'): continue  
        tok = line.strip().split('\t')
        if not tok or tok == ['']: #If it is empty line
            if len(tokens)>1:
                yield tokens
                read += 1
            tokens = [root]
            id = 0
        else:
            try:
                if "." in tok[0] or "-" in tok[0]: continue
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2] ,tok[3], 
                                         tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1 , tok[7]))
                tokens_read+=1

            except IndexError:
                pass

    #Last sentence
    if len(tokens) > 1:
        yield tokens
    print read, 'sentences read.'
    print tokens_read ,'tokens read'


def write_conll(fn, conll_gen):
    """
    Writes a CoNLL file
    """
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write('\t'.join([str(entry.id), entry.form, entry.lemma, entry.cpos, entry.pos, entry.feats, str(entry.pred_parent_id), entry.pred_relation, '_', '_']))
                fh.write('\n')
            fh.write('\n')



numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()




"""
Looks for multiword expressions in the CoNLL file and creates a lookup table that
allows to reconstruct then the output
"""
def lookup_conll_extra_data(fh):
    
    lookup = {}
    sentence_id = 0
    lookup[sentence_id] = {}
    id_insert_before = 1
    
    for line in fh:
        
        if line.startswith('#'): continue
        tok = line.strip().split('\t')

        if not tok or tok == ['']: #If it is empty line
            sentence_id+=1
            id_insert_before = 1
            lookup[sentence_id] = {}
        else:
            if "." in tok[0] or "-" in tok[0]:
                lookup[sentence_id][id_insert_before] = line
            else:
                id_insert_before+=1
 
    return lookup
            
"""
dumps the content of the lookup table extracted by lookup_conll_extra_data
into a output conll_path
"""
def dump_lookup_extra_into_conll(conll_path,lookup):
    
    sentence_id = 0
    word_id = 1

    with codecs.open(conll_path) as f_conll:
        lines = f_conll.readlines()

    #DUMPING the content of the file
    f_conll = codecs.open(conll_path,"w")
    
    for line in lines:
        
        tok = line.strip().split('\t')
        if tok == ['']: #If it is empty line
            sentence_id+=1
            word_id = 1
        else:
            if sentence_id in lookup: 
                if word_id in lookup[sentence_id]:
                    f_conll.write(lookup[sentence_id][word_id])
            word_id+=1
        f_conll.write(line)
        
    f_conll.close()


def get_rooted(conll_str):
    """
    Returns a list of [id,ctag,head] of the nodes rooted to 0
    """
    rooted_elements = []
    
    lines = conll_str.split('\n')
    for l in lines:
        ls = l.split('\t')
        try:
            identifier,tag,head = int(ls[UD_ID_COLUMN]),ls[UD_CTAG_COLUMN],int(ls[UD_HEAD_COLUMN])
            if head == DUMMY_ROOT:
                rooted_elements.append((identifier,tag,head))      
        except ValueError:
            pass   
    return rooted_elements
    

def get_new_single_root(lmultiple_rooted):
    """
    Returns the ID of the first VERB rooted to 0 or the leftmost rooted
    element otherwise
    """
    for e in lmultiple_rooted:
        if e[2] == DUMMY_ROOT and e[1] == UD_CTAG_VERB:
                return e[0]     
    return lmultiple_rooted[0][0]
            
"""
"""
def transform_to_single_root(conll_path):
    
    with codecs.open(conll_path) as f_conll:
        sentences = f_conll.read().split('\n\n')
    
    with codecs.open(conll_path,"w") as f_conll:
        
        i=0
        for s in sentences:
            if s == "": continue
            rooted = get_rooted(s)
            if len(rooted) > 1:
                frv = get_new_single_root(rooted)
                for l in s.split('\n'):
                    ls = l.strip().split('\t')   
                    
                    if ls != [''] and not l.startswith("#"): #If it is empty line
                        if ls[UD_HEAD_COLUMN] != "_" and int(ls[UD_HEAD_COLUMN]) == DUMMY_ROOT and int(ls[UD_ID_COLUMN]) != frv:
                            ls[UD_HEAD_COLUMN] = str(frv)
                        
                    f_conll.write('\t'.join(ls)+"\n")
            else:
                f_conll.write(s+"\n") 
            f_conll.write('\n')
            i+=1
        












