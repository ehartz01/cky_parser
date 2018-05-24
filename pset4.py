import sys, re
import nltk
from nltk.corpus import treebank
from collections import defaultdict
from nltk import induce_pcfg
from nltk.grammar import Nonterminal
from nltk.tree import Tree
from math import exp, pow, log

unknown_token = "<UNK>"  # unknown word token.

""" Removes all function tags e.g., turns NP-SBJ into NP.
"""         
def RemoveFunctionTags(tree):
    for subtree in tree.subtrees():  # for all nodes of the tree
        # if it's a preterminal node with the label "-NONE-", then skip for now
        if subtree.height() == 2 and subtree.label() == "-NONE-": continue
        nt = subtree.label()  # get the nonterminal that labels the node
        labels = re.split("[-=]", nt)  # try to split the label at "-" or "="
        if len(labels) > 1:  # if the label was split in two e.g., ["NP", "SBJ"]
            subtree.set_label(labels[0])  # only keep the first bit, e.g. "NP"

""" Return true if node is a trace node.
"""         
def IsTraceNode(node):
    # return true if the node is a preterminal node and has the label "-NONE-"
    return node.height() == 2 and len(node) == 1 and node.label() == "-NONE-"

""" Deletes any trace node children and returns true if all children were deleted.
"""
def RemoveTraces(node):
    if node.height() == 2:  # if the node is a preterminal node
        return False  # already a preterminal, cannot have a trace node child.
    i = 0
    while i < len(node):  # iterate over the children, node[i]
        # if the child is a trace node or it is a node whose children were deleted
        if IsTraceNode(node[i]) or RemoveTraces(node[i]): 
            del node[i]  # then delete the child
        else: i += 1
    return len(node) == 0  # return true if all children were deleted
    
""" Preprocessing of the Penn treebank.
"""
def TreebankNoTraces():
    tb = []
    for t in treebank.parsed_sents():
        if t.label() != "S": continue
        RemoveFunctionTags(t)
        RemoveTraces(t)
        t.collapse_unary(collapsePOS = True, collapseRoot = True)
        t.chomsky_normal_form()
        tb.append(t)
    return tb
        
""" Enumerate all preterminal nodes of the tree.
""" 
def PreterminalNodes(tree):
    for subtree in tree.subtrees():
        if subtree.height() == 2:
            yield subtree
    
""" Print the tree in one line no matter how big it is
    e.g., (VP (VB Book) (NP (DT that) (NN flight)))
"""         
def PrintTree(tree):
    if tree.height() == 2: return "(%s %s)" %(tree.label(), tree[0])
    return "(%s %s)" %(tree.label(), " ".join([PrintTree(x) for x in tree]))

def BuildVocab(training_set):
    final_v = set()
    first = set()
    for tree in training_set:
        for node in PreterminalNodes(tree):
            if node[0] not in final_v and node[0] in first:
                final_v.add(node[0])
            if node[0] not in first and node[0] not in final_v:
                first.add(node[0])
    final_v.add(unknown_token)
    return final_v

def PreprocessText(corpus, vocabulary):
    final_set = []
    for tree in corpus:
        new_tree = tree
        for node in PreterminalNodes(new_tree):
            if node[0] not in vocabulary:
                node[0] = unknown_token
        final_set.append(new_tree)
    return final_set

def countNP(prods):
    rules = set()
    rulecounts = defaultdict(float)
    for prod in prods:
        if prod.lhs() == Nonterminal("NP"):
            if prod not in rules:
                rules.add(prod)
            rulecounts[prod] += 1.0
    print len(rulecounts)
    most_prob_prods = []
    for count in range(0,10):
        most_prob_prods.append(max(rulecounts, key=lambda i: rulecounts[i]))
        rulecounts[max(rulecounts, key=lambda i: rulecounts[i])] = -1.0
    print most_prob_prods

class InvertedGrammar:
    def __init__(self, pcfg):
        self._pcfg = pcfg
        self._r2l = defaultdict(list)  # maps RHSs to list of LHSs
        self._r2l_lex = defaultdict(list)  # maps lexical items to list of LHSs (but actually to the whole production)
        self.BuildIndex()  # populates self._r2l and self._r2l_lex according to pcfg
			
    def PrintIndex(self, filename):
		f = open(filename, "w")
		for rhs, prods in self._r2l.iteritems():
			f.write("%s\n" %str(rhs))
			for prod in prods:
				f.write("\t%s\n" %str(prod))
			f.write("---\n")
		for rhs, prods in self._r2l_lex.iteritems():
			f.write("%s\n" %str(rhs))
			for prod in prods:
				f.write("\t%s\n" %str(prod))
			f.write("---\n")
		f.close()
        
    def BuildIndex(self):
        """ Build an inverted index of your grammar that maps right hand sides of all 
        productions to their left hands sides.
        """
        for prod in self._pcfg.productions():
            if (prod.is_nonlexical()):
                self._r2l[prod.rhs()].append(prod)
            if prod.is_lexical():
                self._r2l_lex[prod.rhs()].append(prod)
    def Parse(self, sent):
        """ Implement the CKY algorithm for PCFGs, populating the dynamic programming 
        table with log probabilities of every constituent spanning a sub-span of a given 
        test sentence (i, j) and storing the appropriate back-pointers. 
        """
        sent.append(" ")
        dynamic_table = defaultdict(float)
        backpointers = defaultdict(tuple)
        #for j,token in enumerate(sent):
        for j in range(0,len(sent)):
            for rule in self._r2l_lex[(sent[j],)]:
                dynamic_table[(j,j+1,rule.lhs())] = log(rule.prob())
            
            for i in range(j-1,-1,-1):
                for k in range(i+1, j):
                    newlist1 = []
                    newlist2 = []
                    for key in dynamic_table.keys():
                        if key[0] == i and key[1] == k:
                            newlist1.append(key[2])
                        if key[0] == k and key[1] == j:
                            newlist2.append(key[2])
                    for b in newlist1:
                        for c in newlist2:
                            rulelist = self._r2l[(b,c)]
                            for rule in rulelist:
                                if (i,j,rule.lhs()) not in dynamic_table.keys() or dynamic_table[(i,j,rule.lhs())] < log(rule.prob()) + dynamic_table[(i,k,rule.rhs()[0])] + dynamic_table[(k,j,rule.rhs()[1])]:
                                    dynamic_table[(i,j,rule.lhs())] = log(rule.prob()) + dynamic_table[(i,k,rule.rhs()[0])] + dynamic_table[(k,j,rule.rhs()[1])]
                                    backpointers[(i,j,rule.lhs())] = (k,rule.rhs()[0],rule.rhs()[1])
        if sent == ["Terms", "were", "n't", "disclosed", ".", " "]:
            print dynamic_table[(0,len(sent)-1,Nonterminal("S"))]
        return self.BuildTree(dynamic_table,sent,backpointers,(0,len(sent)-1,Nonterminal("S")))
        
    def BuildTree(self, cky_table, sent, backpointers, current):
        """ Build a tree by following the back-pointers starting from the largest span 
        (0, len(sent)) and recursing from larger spans (i, j) to smaller sub-spans 
        (i, k), (k, j) and eventually bottoming out at the preterminal level (i, i+1).
        """
        if (0,len(sent)-1,Nonterminal("S")) not in cky_table:
            return None
        if current in backpointers:
            state = backpointers[current]
            
            left = (current[0],state[0],state[1]) #i k B
            tree_left = self.BuildTree(cky_table,sent,backpointers,left) #follow B

            right = (state[0],current[1],state[2]) #k j C
            tree_right = self.BuildTree(cky_table,sent,backpointers,right) #follow C
            return Tree(str(current[2]),[tree_left,tree_right]) #put the two together
        else: #handle leaf nodes
            return Tree(str(current[2]),[sent[current[1]-1]])





def main():
    treebank_parsed_sents = TreebankNoTraces()
    training_set = treebank_parsed_sents[:3000]
    test_set = treebank_parsed_sents[3000:]

    # Transform the data sets by eliminating unknown words.
    
    vocabulary = BuildVocab(training_set)
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)
    print "first training set tree:"
    print PrintTree(training_set_prep[0])
    print "first test set tree:"
    print PrintTree(test_set_prep[0])
    """ Implement your solutions to problems 2-4.
    """
    prods = []
    for tree in training_set_prep:
        for prod in tree.productions():
            prods.append(prod)
    countNP(prods)
    pcfg = induce_pcfg(Nonterminal("S"), prods)

    ig = InvertedGrammar(pcfg)
    ig.PrintIndex("index.txt")
    print ig.Parse(["Terms", "were", "n't", "disclosed", "."])
    print ["Terms", "were", "n't", "disclosed", "."]

    bucket1 = []
    bucket2 = []
    bucket3 = []
    bucket4 = []
    bucket5 = []
    buckets = [bucket1, bucket2, bucket3, bucket4, bucket5]
    for tree in test_set_prep:
        if len(tree.leaves()) < 10:
            bucket1.append(tree)
        if len(tree.leaves()) >= 10 and len(tree.leaves()) < 20:
            bucket2.append(tree)
        if len(tree.leaves()) >= 20 and len(tree.leaves()) < 30:
            bucket3.append(tree)
        if len(tree.leaves()) >= 30 and len(tree.leaves()) < 40:
            bucket4.append(tree)
        if len(tree.leaves()) >= 40:
            bucket5.append(tree)
    for count, bucket in enumerate(buckets):
        print "bucket" + str(count+1) + " has " + str(len(bucket)) + " sentences."
    for count, bucket in enumerate(buckets):
        f = open("test_" + str(count+1), "w")
        for tree in bucket:
            tree.un_chomsky_normal_form()
            reparsed = ig.Parse(tree.leaves())
            if reparsed != None:
                reparsed.un_chomsky_normal_form()
                f.write(PrintTree(reparsed) + "\n")
            else:
                f.write("\n")
        f.close()
        f = open("gold_" + str(count+1), "w")
        for tree in bucket:
            f.write(PrintTree(tree) + "\n")
        f.close()
        

if __name__ == "__main__": 
    main()



