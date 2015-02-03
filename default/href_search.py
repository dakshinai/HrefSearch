"""
targets = [
    ('a', 'href'), ('applet', 'codebase'), ('area', 'href'), ('base', 'href'),
    ('blockquote', 'cite'), ('body', 'background'), ('del', 'cite'),
    ('form', 'action'), ('frame', 'longdesc'), ('frame', 'src'),
    ('head', 'profile'), ('iframe', 'longdesc'), ('iframe', 'src'),
    ('img', 'longdesc'), ('img', 'src'), ('img', 'usemap'), ('input', 'src'),
    ('input', 'usemap'), ('ins', 'cite'), ('link', 'href'),
    ('object', 'classid'), ('object', 'codebase'), ('object', 'data'),
    ('object', 'usemap'), ('q', 'cite'), ('script', 'src'), ('audio', 'src'),
    ('button', 'formaction'), ('command', 'icon'), ('embed', 'src'),
    ('html', 'manifest'), ('input', 'formaction'), ('source', 'src'),
    ('video', 'poster'), ('video', 'src'),
]
"""

#ethical requests and limit to 1 page per host - priority index
#spider traps - priority index
#find useful pages first - rate of change, quality and usefulness - priority index

#statistics not done - parent child gen 
#no retries
#scope with dictionary or some algo for focussed
#remove default files and reduce .. parent path
#multithreading
#check for freshness if url occurs again in the crawl. Will be useful to update a stored index
#check same content seen
#checkpoint the crawler
#url to fixed names


import urllib2
import urllib
import bs4
import pprint
import re
import urlparse
from collections import deque
import time
import nltk
from nltk.stem import PorterStemmer
import cProfile
import pstats
import cPickle
from collections import defaultdict
import math
import networkx as nx  
from networkx.algorithms.components.weakly_connected import weakly_connected_component_subgraphs 
import scipy 

"""
    word_index-dictionary (string-wordname, postings)
    postings -dictionary (string-docname, wordcount)    
    flipped - inverted copy of index that holds word occurances and count in each document
    N- total number of documents    
    diwords- tfidf of vocab word in documents
    di2dict - sum of tfidf of all vocab words in documents    
    
    cosine - cosine similarity between the tfidf of the document and the raw tf of query  
    tokenquerytf - raw tf of vocab word in query, word not in list implies raw tf is 0     
    """  

seed="http://www.tamu.edu/academics/index.html"
timeout=5
crawl_limit=200
index_file="index"
exclude_types=dict({"application/pdf":0})

tocrawl = deque([seed]) 

 
exclude_list=dict()
stemmer=PorterStemmer()  
                                          
word_index = dict() 
outlinks=dict()
DG=nx.DiGraph()

crawled = dict() 
flipped = defaultdict(dict) 
diwords=dict()
di2dict=dict()   


"""
HITS implementation using eigen vector multiplication
"""
def hitseigenmult(lsg,tol):  
    """n is the number of tweet entities identified in final set for HITS"""
    n=lsg.number_of_nodes()
    """extracting adjacency sparse matrix from directed graph"""
    """an edge from entity A to B implies A is hub of B and B is authority of A"""
    A=nx.to_scipy_sparse_matrix(lsg,nodelist=lsg.nodes())
    """setting hubs and authorities to all ones, vector size: number of tweet entities"""
    h=scipy.ones(n)
    a=scipy.ones(n)
    """Computing A transpose from A"""
    AT=A.transpose()
    count=0  
    
    AAT=A*AT
    ATA=AT*A
    """Computing hubs and authorities (h,a) incrementally. Every iteration uses hubs and authorities from previous iteration (lh,la)"""
    while True:  
        lh=h
        la=a
    
        """hub value of entity is sum of all authorty values of entity"""
        """A [i][j] implies i is hub of j or j is authority of i"""
        """h=A * A transpose * prev h"""
        h=AAT*lh    
        """normalizing h vector"""
        h=h/h.sum()   
        
        """finding the tolerance level of difference of new hub values to old values"""
        err=scipy.absolute(h-lh).sum()
        """we end computation if error falls below tolerance"""
        if err < tol:
            break   
          
        """authority value of entity is sum of all hub values of entity"""   
        """A transpose [i][j] implies i is authority of j or j is hub of i""" 
        """a=A Transpose * A * prev a"""
        a=ATA*la 
        """normalizing a vector""" 
        a=a/a.sum()    
       
        """finding the tolerance level of difference of new hub values to old values"""
        err=scipy.absolute(a-la).sum()
        """we end computation if error falls below tolerance"""
        if err < tol:
            break   
      
        count=count+1
    
    """returning dictionaty of entity name and normalized values"""
    hubs=dict(zip(lsg.nodes(),h/h.sum()))
    auths=dict(zip(lsg.nodes(),a/a.sum()))
    return hubs,auths,count
    
def get_page(url):
    try:        
        
        response = urllib2.urlopen(url, None)  
                
        return response
    except:
        return None
    

def union(crawlQueue, links):
    for e in links:
        if e not in crawlQueue:
            crawlQueue.append(e)

def get_all_links(url,page):  
    links=[]  
    soap=bs4.BeautifulSoup(page)
    
    for link in soap.findAll("a"):
        short_url=link.get("href")
        if short_url and exclude_list.has_key(urlparse.urljoin(url, short_url))==False:    
            recon_url=urlparse.urljoin(url, short_url) 
            recon_url=urllib.unquote_plus(recon_url)
            recon_url=urlparse.urldefrag(recon_url)[0]   
            DG.add_edge(url, recon_url)         
            links.append(recon_url)
    return links

def crawl_web(): 
    while len(tocrawl)>0: 
        
        if(len(crawled)>=crawl_limit):
            break                            
        last_tocrawl_url = tocrawl.popleft()                           
        if  exclude_list.has_key(last_tocrawl_url)==False and crawled.has_key(last_tocrawl_url)==False:      #hash lookup   
            response=get_page(last_tocrawl_url)             
            
            if(response!=None and response.getcode()==200):
                
                contenttype=response.info()["Content-Type"]  
                if exclude_types.has_key(contenttype)==False:                    
                    try:  
                        content =  response.read()  
                                           
                    except:
                        print "Skipping "+last_tocrawl_url                   
                   
                    
                    if content != "":   
                                          
                        add_page_to_index(last_tocrawl_url, content) 
                        DG.add_node(last_tocrawl_url)                   
                        links=get_all_links(last_tocrawl_url,content)  
                        
                        outlinks[last_tocrawl_url]=links                       
                        union(tocrawl,links)                             
                         
                        print len(crawled)+1, last_tocrawl_url               
                        
                        if response.info() and response.info()["Date"]:                                              
                            crawled[last_tocrawl_url]=time.strptime(response.info()["Date"][:25],"%a, %d %b %Y %H:%M:%S")
                        else:
                            crawled[last_tocrawl_url]=time.gmtime()
                        
                
            else:
                print "Skipping "+last_tocrawl_url
        

def add_to_index(keyword, last_tocrawl_url):  
    postings=dict()    
    count=0  
    
    if word_index.has_key(keyword):
        postings=word_index[keyword]
        if postings.has_key(last_tocrawl_url):
            count=postings[last_tocrawl_url]   
                
    postings[last_tocrawl_url]=count+1
    word_index[keyword]=postings
               

def add_page_to_index(last_tocrawl_url, page): 
    
    
    soap=bs4.BeautifulSoup(page)    
    
    content=soap.getText().lower()   
    
    
    stopwords = nltk.corpus.stopwords.words('english')    
    words = set(re.split('\W|\'',content))  
    words.remove("")     
    
    tokens = [w for w in words if not w in stopwords]
    for token in tokens:                                
        #add_to_index(stemmer.stem(token), last_tocrawl_url)
        add_to_index(token, last_tocrawl_url)

def lookup(index, keyword):
    for entry in index:
        if entry[0] == keyword:
            return entry[1]
    return None


try:
    i_file = open(index_file, 'rb')   
    word_index=dict(cPickle.load(i_file))      
    outlinks=dict(cPickle.load(i_file))
    DG=nx.DiGraph(cPickle.load(i_file))
    crawled=dict(cPickle.load(i_file))
    flipped=dict(cPickle.load(i_file))
    diwords=dict(cPickle.load(i_file))
    di2dict=dict(cPickle.load(i_file))    
    i_file.close()
        
    
except:
    print "No previous index found" 
    

    
if(len(word_index)==0 or len(outlinks)==0 or len(crawled)==0):
    print 'Building index..'
    start = time.clock()
    crawl_web()
    print "Crawl time:"+str(time.clock() - start)+" seconds"  
    
    wf=open('word_index_text','w') 
    pprint.pprint(word_index, wf)  
    
    for key, val in word_index.items():            
        for subkey, subval in val.items(): 
            flipped[subkey][key] = subval
    
    N= len(flipped)  
      
    for doc, docworddict in flipped.iteritems():        
            sum2tfidfalld=0
            for word, tf in docworddict.iteritems():
                idfw=math.log10(N*1.0/len(word_index[word]))            
                tfidftd=(1+math.log10(tf))*idfw
                sum2tfidfalld=sum2tfidfalld+math.pow(tfidftd, 2)
                if(diwords.has_key(doc)):
                    diword=diwords[doc]
                else:
                    diword=dict()
                    
                diword[word]=tfidftd
                diwords[doc]=diword          
            
                    
            di2dict[doc]=sum2tfidfalld 
        
    i_file = open(index_file, 'wb')   
    cPickle.dump(word_index,i_file)     
    cPickle.dump(outlinks, i_file)
    cPickle.dump(DG, i_file)
    cPickle.dump(crawled, i_file)
    cPickle.dump(flipped, i_file)
    cPickle.dump(diwords, i_file)   
    cPickle.dump(di2dict, i_file)       
    i_file.close()
    
    print "Index built in "+str(time.clock() - start)+" seconds" 
        
#cProfile.run('word_index= crawl_web()','stats')
#p = pstats.Stats('stats')
#p.strip_dirs().sort_stats('cumtime').print_stats()   

wf=open('flipped_text','a') 
for key, val in flipped.items():
    pprint.pprint(key,wf)
    pprint.pprint(val, wf)

wf=open('diwords_text','w') 
pprint.pprint(diwords, wf)
wf=open('di2dict_text','w') 
pprint.pprint(di2dict, wf)

   
"""finding the largest weakly connected subgraph and save it as 

dsg"""
 
         
wcc=weakly_connected_component_subgraphs(DG)
maxsize=0
lsg=nx.DiGraph()
for sg in wcc:
    if(sg.size()>maxsize):
        maxsize=sg.size()
        lsg=sg


if(len(lsg)>0):
    tol=1.0e-12   
    hubs,auths,iters=hitseigenmult(lsg, tol)
    
    print "No. of iterations by eigen mult-"+str(iters)
    print ""
      
    
    print "Hubs and scores-------------------------------"
    print ""
    
    """Displaying top 20 hubs"""
    
    c=20
    count=c
    for hub in sorted(hubs, key=hubs.get, reverse=True):
        if(count>0):
            print hub, hubs[hub]
        count=count-1
    
    print ""
    print "Authorities and scores------------------------"  
    print ""
    
    """Displaying top 20 authorities"""
    
    count=c
    for auth in sorted(auths, key=auths.get, reverse=True):
        if(count>0):
            print auth, auths[auth]
        count=count-1    
         
       
N= len(flipped) 
        
while 1:            
    raw_query=raw_input('Enter your raw query:')  
    
    if(raw_query.lower()=="exit"):
        print "Exiting.."
        break
    else:    
        
        
        start = time.clock()
        
        print "Boolean Retrieval.."
        matched_docs=set();       
        isset1=1    
        tokenquerytf=dict() 
        cosine=dict() 
        
        tokens=list(re.split('\W|_',raw_query.lower()))  
       
        """
        Computing the intersection of documents containing each token 

as we perform AND search
        """
        for token in tokens:        
            if(token!=""):                  
                if(word_index.has_key(token)):                     
                    if(isset1==1):
                        matched_docs=set(word_index[token].keys()) 
                        isset1=0
                    else:
                        matched_docs=set(word_index[token].keys

()).intersection(matched_docs)  
                    """If result set is reduced to none, skip 

checking for other tokens"""              
                    if(len(matched_docs)==0):
                        break
                else:
                    """If any token is not found in any document, 

return none"""
                    matched_docs.clear()
                    break;
           
        if(len(matched_docs)>0):        
            for matched_doc in matched_docs:
                print matched_doc
                '''
                if(str(matched_doc).find("\\")!=-1):
                    print matched_doc.split('\\')[-1].split(".txt")

[0]
                else:
                    print matched_doc.split('/')[-1].split(".txt")[0]
                    '''
            print "Boolean Retrieval in "+str(time.clock() - start)+" seconds" 
        else:        
            print "Sorry no match found using Boolean Retrieval!"     
        
        start = time.clock()
        
        print "Vector Space Retrieval.." 
        
        """Compute number of occurances of each word in query in 

query"""
        
        for token in tokens:
            if(tokenquerytf.has_key(token)):
                tokenquerytf[token]=tokenquerytf[token]+1
            else:
                tokenquerytf[token]=1   
        
       
        """
        Compute cosine similarity terms between the tfidf of the 

document and the raw tf of query    
        for evry doc in cosine
            for every token in query
                find tf of token in query
                find tfidf of token in doc as doc,token
                
            compute sum of prod of tf and tfidf and sum of square of 

tf
            divide sum by root of sum of square of tf
            divide by root of sum of docwordtfidf^2    
        """  
        
        for doc, docworddict in flipped.iteritems():
            """defaulting for all docs that do not contain any of 

query terms"""
            cosine[doc]=0  
            sumqidi=0
            qi2=0
            for token in tokens:            
                qi=tokenquerytf[token]
                qi2=qi2+math.pow(qi, 2)
                diword=dict(diwords[doc])
                if(diword.has_key(token)):
                    di=diword[token]
                else:
                    di=0
                    
                sumqidi=sumqidi+qi*di                
                
            di2=di2dict[doc]
            
            
            if(sumqidi==0 or qi2==0 or di2==0):
                """skipping all docs that have no weights"""                
                cosine[doc]=0 
            else:
                cosine[doc]=sumqidi/(math.sqrt(qi2)*math.sqrt(di2))  
            
        """Sorting cosine values of each document in descending order 

and returning the top 50"""
        
        pcount=0   
        for doc in sorted(cosine, key=cosine.get, reverse=True):
            if(cosine[doc]>0):
                pcount=pcount+1
                if(pcount>50):
                    break   
                print doc , "["+str(cosine[doc])+"]"           
                '''if(str(doc).find("\\")!=-1):
                    print doc.split('\\')[-1].split(".txt")[0] +" 

["+str(cosine[doc])+"]"   
                else:
                    print doc.split('/')[-1].split(".txt")[0] +" 

["+str(cosine[doc])+"]" '''     
          
        if(pcount>0):
            print "Vector Retrieval in "+str(time.clock() - start)+" seconds"  
        elif(len(matched_docs)==N and N!=0):
            print "Query found in all documents (Refer: Boolean Retrieval results)" 
        else:   
            print "Sorry no match found using Vector Space Retrieval!"    
            
            

