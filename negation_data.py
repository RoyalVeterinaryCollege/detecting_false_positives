import pandas as pd
import gensim
import numpy as np


def reload2():
    exec(open("negation_data.py").read(), globals())

data_root_folder = 'data'
data_raw_folder = data_root_folder + '/raw'
data_processed_folder = data_root_folder + '/processed'
data_cached_folder= data_root_folder + '/cached'
randomisation_salt = 'iodfl'

# the study labels which are 1.0 labels, other labels are 0.0
study_class_positive_labels = {
'E3ACC57B-584F-4290-92DE-32242526EF77'.lower():set(['IMHA Case - confirmed']),
'F6FD297B-7DBB-40F7-90D5-4837C320570F'.lower():set(['Is a KCS case']),
'C580B5CC-BA58-4BF1-8CCA-7673D7B47D10'.lower():set(['Case']),
'5B4196D1-ECBF-4895-9CFF-822B84203509'.lower():set(['Is an OA case']),
'4BB814FE-5F8D-417C-A015-9C60767DB9C6'.lower():set(['Case']),
'8425EC75-183E-4281-ADB0-BA5D27947BD3'.lower():set(['Is a demodicosis case']),
'B0A61ACD-B4CB-4F54-B294-95EFFB485245'.lower():set(['DM Case - Confirmed']),
'057ECDCB-64FC-4F50-AC86-70EEA72301FF'.lower():set(['Anal sac case']),
'1BC06999-E70D-42F1-AC01-5D7EF6D8DF4B'.lower():set(['CKD case  - confirmed']),
'BC030E37-1838-473C-A854-1300FEBCA9E1'.lower():set(['HAC case - confirmed']),
'85D2A38E-E0AE-464F-8D5A-001B4D303E15'.lower():set(['This dog is an EJD case']),
'4FC3C861-922D-4E9B-AF96-9B224E5952D9'.lower():set(['Dystocia case - confirmed']),
'71D6DF39-943C-4174-865C-0540F72D6C2A'.lower():set(['Pat lux case - confirmed']),
'314C0838-3F0A-4A06-8AAE-C3AEFA581BF6'.lower():set(['AUI Case - confirmed']),
'4E278045-23D2-46E7-9C39-748B9600D694'.lower():set(['Hypertensive']),
'F3D004BD-726B-4EE9-B23C-74FE26C242D6'.lower():set(['Case RTA - confirmed']),
'0ED9F72A-5B6B-4296-9743-411675D6CD4D'.lower():set(['Spey case']),
'C879AC8F-A8EE-4F40-B665-7FAEF159746C'.lower():set(['Dental case'])
}


# sentence corpus header row
header = 'StudyId,PatientID,NoteId,CaseLabel,Sublabel,SentenceLabel,DiagnosisDate,SourceNoteRecordedDate,Sentence\n'

def get_referral_study_phrases():
    dict={}
    dict['43']='eye discharge,ocular discharge'
    dict['62']='ataxia'
    dict['65']='lame,lameness'
    dict['74']='pale mm,pale mucous membranes'
    dict['79']='panting'
    dict['84']='Pruritus'
    dict['86']='Regurgitation,regurgitating'
    dict['90']='hypersalivation,salivation,drool,drooling'
    dict['93']='seizure,seizures'
    dict['96']='sneeze,sneezing'
    dict['98']='Traumatic episode'
    dict['103']='weakness'
    dict['178']='reduced appetite'
    dict['197']='Weight loss'
    dict['312']='Anorexia,inappetence'
    dict['350']='Acromegaly'
    dict['396']='anal sac disorder'
    dict['414']='Ascites'
    dict['425']='behaviourist'
    dict['466']='burn injury'
    dict['506']='dilated cardiomyopathy'
    dict['511']='Cardiomyopathy'
    dict['518']='cardiac arrest,arrest'
    dict['528']='cauda equina'
    dict['529']='cellulitis'
    dict['560']='cleft palate'
    dict['562']='coagulopathy'
    dict['565']='b12,cobalamin'
    dict['567']='senile,Cognitive dysfunction'
    dict['579']='Congenital disorder'
    dict['583']='Constipation'
    dict['593']='Corneal perforation'
    dict['594']='Corneal scar,corneal scarring'
    dict['624']='Cystitis'
    dict['625']='bacterial cystitis'
    dict['631']='broken tooth'
    dict['648']='ringworm,ring worm'
    dict['668']='Distichiasis'
    dict['676']='ear infection'
    dict['684']='elbow dysplasia'
    dict['692']='empyema'
    dict['708']='Entropion'
    dict['718']='idiopathic epilepsy'
    dict['740']='asthma'
    dict['829']='tibial fracture'
    dict['889']='gastric ulcer'
    dict['891']='gastritis'
    dict['895']='Gastroenteritis'
    dict['921']='haemoperitoneum,haemoabdomen'
    dict['952']='heat stroke'
    dict['963']='Hepatitis'
    dict['967']='hepatopathy'
    dict['975']='hiatal hernia'
    dict['979']='umbilical hernia'
    dict['991']='horner\'s syndrome,horners syndrome'
    dict['999']='bee sting,wasp sting,hymenoptera'
    dict['1014']='Hyperlipidaemia'
    dict['1025']='allergic skin disease'
    dict['1037']='hyperthyroidism'
    dict['1045']='Hypoglycaemia'
    dict['1046']='Hypokalaemia'
    dict['1060']='iohc'
    dict['1063']='ibd,Inflammatory bowel disease'
    dict['1074']='Intervertebral disc disorder'
    dict['1096']='rodenticide'
    dict['1097']='chocolate'
    dict['1150']='kcs,dry eye,keratoconjunctivitis sicca'
    dict['1153']='laryngeal paralysis'
    dict['1228']='adrenal mass'
    dict['1287']='cutaneous mass'
    dict['1290']='splenic mass lesion'
    dict['1313']='Megaoesophagus'
    dict['1338']='mrsa'
    dict['1360']='myelitis'
    dict['1473']='splenic tumour'
    dict['1503']='Nuclear sclerosis'
    dict['1534']='elbow osteoarthritis'
    dict['1577']='ot ext,Otitis externa,oe'
    dict['1609']='lungworm'
    dict['1610']='Angiostrongylosis'
    dict['1624']='ear mite,ear mites'
    dict['1658']='Periodontal disease'
    dict['1678']='Pleural effusion'
    dict['1683']='aspiration pneumonia'
    dict['1685']='bronchopneumonia'
    dict['1751']='pra,Progressive retinal atrophy'
    dict['1760']='pseudopregnancy'
    dict['1776']='Pulmonic valve stenosis'
    dict['1779']='Pyelonephritis'
    dict['1785']='Pyometra'
    dict['1786']='PUO'
    dict['1840']='soft tissue sarcoma'
    dict['1849']='Sepsis'
    dict['1913']='epilepticus'
    dict['1914']='Stenotic nares'
    dict['1937']='Synovitis'
    dict['1938']='Syringomyelia'
    dict['1964']='Tetanus'
    dict['2019']='hairball,trichobezoar'
    dict['2031']='ectopic ureter'
    dict['2075']='juvenile vaginitis'
    dict['2083']='vestibular episode'
    dict['2106']='petechiation'
    dict['2127']='infected wound'
    dict['2134']='blue eye'
    dict['2151']='dysphagia'
    dict['2153']='aural abnormality,ear abnormality'
    dict['2162']='haematochezia,bloody diarrhoea'
    dict['2164']='melaena'
    dict['2169']='gagging,retching'
    dict['2176']='head tilt'
    dict['2187']='Lethargy'
    dict['2206']='depressed'
    dict['2228']='Nystagmus'
    dict['2242']='paralysis,paresis'
    dict['2255']='polyphagia,ravenous'
    dict['2263']='anisocoria'
    dict['2284']='Tachypnoea'
    dict['2288']='tremors,shaking,trembling,twitching'
    dict['2292']='Vocalisation'
    dict['2899']='neck pain'
    dict['4334']='renal disorder,kidney disorder'
    dict['4380']='Systolic dysfunction'
    dict['4418']='Systemic hypertension,hypertension'
    dict['4566']='SVT,Supraventricular tachycardia'
    dict['4742']='aggressive,aggression,growling'
    dict['4756']='skin lesions'
    dict['5399']='skin disorder'
    dict['5400']='alopecia'
    dict['5788']='Muscle injury'
    dict['5862']='hyperaldosteronism'
    dict['7028']='chiari,chiari like,chiari-like'
    dict['7514']='acrochordon'
    dict['7516']='aural discharge,ear discharge'
    dict['7550']='underweight'
    dict['7573']='stiffness'
    dict['7580']='Corneal abnormality'
    dict['7587']='blepharospasm'
    dict['7604']='Halitosis'
    dict['7627']='Stridor'
    dict['7637']='icterus,jaundice'
    dict['7890']='head tremors'
    dict['9667']='parvo,parvovirus'
    dict['11013']='itp'
    dict['15219']='lily poisoning'
    dict['17179']='infiltrative lipoma'
    dict['17981']='mue'
    dict['18779']='hit by car'
    dict['19783']='overweight'
    dict['20267']='colitis'

    for (key,value) in dict.items():
        lower = value.lower()
        lower = lower.split(',')
        phrases = list()
        for token in lower:
            phrases.append(r'\b%s\b' % token)
        dict[key]=phrases
    return dict

def get_study_phrases():
    # todo fuzzy matching or clean up sentences, this approach will lack specificity as it doesnt handle noisey corpus
    # todo are these even the right words?!  need to check with dan
    study_phrases=dict()
    study_phrases['f6fd297b-7dbb-40f7-90d5-4837c320570f'.lower()]=list([r'\bkeratoconj[a-z]*\b',r'\bkcs[a-z]*\b',r'\bdry[ -]eye\b',r'\bsicca[a-z]*\b',r'\boptimmune\b',r'\bcyclosp[a-z]*\b',r'\btacro[a-z]*\b'])

    study_phrases['E3ACC57B-584F-4290-92DE-32242526EF77'.lower()] = list([r'\bhaemolytic\b',r'\bhaemolys[a-z]*\b',r'\bimha\b',r'\baiha\b'])

    study_phrases['8425EC75-183E-4281-ADB0-BA5D27947BD3'.lower()] = list([r'\bdemod[a-z]*\b',r'\bdemodi[a-z]*\b',r'\bdemodicosis\b',r'\bdemodectic\b']) #demod* demodex* demodectic* demodicosis*
    # OA osteoa* DJD joint dise*, osteoph* arth*
    study_phrases['5b4196d1-ecbf-4895-9cff-822b84203509'.lower()] = list([r'\boa\b',r'\bosteoa[a-z]*\b',r'\bdegen[a-z]*\b[ -]*joint[a-z]*\b([ -]*disease)?',r'joint dise[a-z]*\b',r'\bdjd\b',r'\bosteoph[a-z]*\b',r'\barth[a-z]*\b'])

    study_phrases['c580b5cc-ba58-4bf1-8cca-7673d7b47d10'.lower()] = list([r'\bspondylosis\b',r'\bdis[ck][ -]disea[a-z]*\b',r'\bdis[ck][ -]dz\b',r'\bdis[ck][ -]dz\b',r'\bintervert[a-z]*\b dis[ck] dis[a-z]*\b',r'\bspinal pain',r'\bparesis\b',r'\bivdd\b',r'\bdis[ck] prolapse']) # these come from selected txt on problem report as I don't knwo what the search terms were.  There a re alot more terms that could be added here.
    
    # cruciate ccl cranial draw acl tta tplo  lateral suture extrcapsular suture  de angelis suture
    study_phrases['4BB814FE-5F8D-417C-A015-9C60767DB9C6'.lower()] = list([r'\bcruciate\b',r'\bccl\b',r'\bcranial[ -]draw\b',r'\bacl\b',r'\btta\b',r'\btplo\b',r'\blateral[ -]sutures?\b',r'\bextracapsular[ -]sutures?\b',r'\bde[ -]angelis[ -]sutures?\b'])
    study_phrases['B0A61ACD-B4CB-4F54-B294-95EFFB485245'.lower()] = list([r'\binsul[a-z]*\b',r'\bdm\b',r'\bdiabe[a-z]*\b',r'\bmellitus\b',r'\bdka\b',r'\bketoacid[a-z]*\b',r'\bhyperglyc[a-z]*\b',r'\bglucosur[a-z]*\b',r'\bketonur[a-z]*'])
    study_phrases['057ECDCB-64FC-4F50-AC86-70EEA72301FF'.lower()] = list([r'\bags?\b',r'\beags?\b',r'\bag\'s\b',r'\banal g[a-z]*\b',r'\banal\b[ -]saccu[a-z]*\b',r'\bxag\'?s\b',r'\be?xpress\b[ -]\banal\b'])
    study_phrases['1BC06999-E70D-42F1-AC01-5D7EF6D8DF4B'.lower()] = list([r'\bk ?[-\\\/ ] ?d\b',r'\bkd\b',r'\bkidney\b',r'\brenal\b',r'\bckd\b',r'\bckf\b',r'\bcrd\b',r'\bcrf\b']) 
    study_phrases['BC030E37-1838-473C-A854-1300FEBCA9E1'.lower()] = list([r'\bpdh\b',r'\badh\b',r'\bhyperadren[a-z]*\b',r'\bpituit[a-z]\b',r'\badrenal\b',r'\bcushing ?\'?s',r'\badrenalec[a-z]*\b',r'\badrenomeg[a-z]*\b'])
    study_phrases['85D2A38E-E0AE-464F-8D5A-001B4D303E15'.lower()] = list([r'\belbow[a-z]*\b',r'\bosteo[a-z]*\b',r'\barth[a-z]*\b',r'\bancon[a-z]*\b',r'\bcoronoid[a-z]*\b',r'\bed\b'])
    study_phrases['4FC3C861-922D-4E9B-AF96-9B224E5952D9'.lower()] = list([r'\bdysto[a-z]*\b',r'\bdisto[a-z]*\b'])
    study_phrases['71D6DF39-943C-4174-865C-0540F72D6C2A'.lower()] = list([r'\bpatel[a-z]*\b',r'\bslipping[ -]knee[ -]?cap[a-z]*\b',r'\blux[a-z]*\b'])
    study_phrases['314C0838-3F0A-4A06-8AAE-C3AEFA581BF6'.lower()] = list([r'\bincont[a-z]*\b',r'\busmi\b',r'\bincompet[a-z]*\b',r'\burin[a-z]*\b',r'\burethral sp[a-z]*\b',r'\bnocturia\b',r'\bwetting\b',r'\bwet the bed\b',r'\bdribbling urin[a-z]*\b',r'\bleaking\b'])

    # this study coded patients on these study lists, so these are teh tokens we will use
    study_phrases['4E278045-23D2-46E7-9C39-748B9600D694'.lower()] = list([r'\bblood[ -]pressure\b',r'\bamlod[a-z]*\b',r'\bbp\b',r'\bhyperten[a-z]*\b',r'\bhyphaema[a-z]?\b',r'\bistin\b',r'\bretin[a-z]*\b[ -]detach[a-z]*\b',r'\bretin[a-z]*\b[ -]haem[a-z]*\b'])
    
    study_phrases['F3D004BD-726B-4EE9-B23C-74FE26C242D6'.lower()] = list([r'\bhit\b',r'\btra\b',r'\brtc\b',r'\br[au]n[ -]over\b',r'\bknock\b',r'\btraffic[ -]collision\b',r'\bvehicle[a-z]*\b',r'\bcar\b']) 
    study_phrases['0ED9F72A-5B6B-4296-9743-411675D6CD4D'.lower()] = list([r'\bsp[ea]y\b',r'\bovario[a-z]*\b',r'\bove\b',r'\bovh\b',r'\bohe\b',r'\bovar[a-z]*\b'])

    # this one was so large I entered it programmatically
    dental_search = "perio dent dont toot teet oral extract mandib maxil mouth ging Ging tart tartar EENT plaq calc arcade apex apic root crown incisors molar canine pml carnas furc halit gum buccal lingual mesial palat labial lip commis brush epul stoma caries enam".split()
    dental_phrases=list()
    for token in dental_search:
        dental_phrases.append(r'\b%s[a-z]*\b' % token.lower())

    dental_phrases.append(r'\bd ?[-\\\/ ] ?t\b')
    dental_phrases.append(r'\bt ?[-\\\/ ] ?d\b')
    study_phrases['C879AC8F-A8EE-4F40-B665-7FAEF159746C'.lower()] = dental_phrases
    referral_phrases = get_referral_study_phrases()
    study_phrases.update(referral_phrases)

    return study_phrases



class ConcatenatedDoc2Vec(object):
    """
    Concatenation of multiple models for reproducing the Paragraph Vectors paper.
    Models must have exactly-matching vocabulary and document IDs. (Models should
    be trained separately; this wrapper just returns concatenated results.)
    """
    def __init__(self, models):
        self.models = models
        if hasattr(models[0], 'docvecs'):
            self.docvecs = ConcatenatedDocvecs([model.docvecs for model in models])

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])

    def __contains__(self, key):
        return any([key in model for model in self.models])

    def infer_vector(self, document, alpha=0.1, min_alpha=0.0001, steps=5):
        return np.concatenate([model.infer_vector(document, alpha, min_alpha, steps) for model in self.models])

    def train(self, ignored):
        pass  # train subcomponents individually



def load_word2vec_sg():
    return gensim.models.KeyedVectors.load_word2vec_format('data/processed/vec_sg_20180321.txt', binary=False, unicode_errors='ignore')

def load_word2vec_combined():
    sg = load_word2vec_sg()
    cbow = gensim.models.KeyedVectors.load_word2vec_format('data/processed/vec_cbow.txt', binary=False)
    combined = ConcatenatedDoc2Vec([sg,cbow])
    return combined

# tokenize using nltk but also split any words containing / or \ into 3 words, as nltk doesnt do this
def tokenize(sentence):
    import nltk
    import re
    r= r'[a-z][\./\\-][a-z]'

    match = re.search(r, sentence)
    if match:
        sentence = sentence[0:match.start() + 1] + ' ' + sentence[match.start()+1] + ' ' + sentence[match.end()-1:]
        # and recurse to look for more matches
        return tokenize(sentence)
    else:
        return nltk.tokenize.word_tokenize(sentence)


# produces a label 1/0 for a sentence based on the study & note meta data : primary dataset
def get_sentence_label_primary_dataset(StudyId,Label,Sublabel,DiagnosisDate,SourceNoteRecordedDate):
    # rules : 
    # classlabel = negative then 0
    # sublabel = pre-existing then 1
    # else SourceNoteRecordedDate < DiagnosisDate then 0 else 1

    # is it a negative class label?  (all class labels which are not the postive label are negative)
    positive_label = Label in study_class_positive_labels[StudyId.lower()] 
    if not positive_label: return 0

    # pre-existing all postive
    if Sublabel == 'Pre-existing': return 1
    if Sublabel == 'Unknown if incident or pre-existing': return 1 # todo : not sure this is safe but it's reasonable?

    from dateutil import parser
    
    if pd.isnull(DiagnosisDate): return 1 # missing diagnosis date, just assume 1

    try:
        dd = parser.parse(DiagnosisDate[:-3],dayfirst=True)
    except TypeError:
        return 1 # missing diagnosis date, just assume 1

    nd = parser.parse(SourceNoteRecordedDate[:-3],dayfirst=True)
    if nd >= dd: return 1
    
    return 0



# laods the ConText feature regexes
def load_context_feature_definitions():
    from sklearn import preprocessing
    context_features_ordinals = list() # tuples : regex , feature ordinal, feature name
    context_classes = dict()  # Type -> ordinal .  the kinds of things context recognises , indictation, DEFINITE_NEGATED_EXISTENCE etc
    context_tsv=pd.read_csv(data_cached_folder + '/' + 'ConText Semantic phrases.csv', sep=',',header=1) # this file was taken from "https://raw.githubusercontent.com/chapmanbe/pyConTextNLP/master/KB/lexical_kb_05042016.tsv"
    

    for index, row in context_tsv.iterrows():
        Lex,Type,Regex,Direction = row

        if Type not in context_classes:
            context_classes[Type]=len(context_classes)

        clean_regex = clean(Regex) # deal with panda nulls

        regex = None
        if clean_regex == "":
            regex="\\b" + Lex + "\\b"
        else:
            regex=clean_regex

        context_features_ordinals.append((regex,context_classes[Type],Type))

    
    lb = preprocessing.LabelBinarizer()
    class_ordinals = list([ordinal for regex, ordinal, feature_type in context_features_ordinals])
    class_ordinals.append(-1) # add a class for 'no context feature', makes it easuer to use the binariser
    lb.fit(class_ordinals)

    regex_features = list()  # regex -> feature vector
  
    for regex, ordinal,feature_type in context_features_ordinals:
        regex_features.append((regex, lb.transform([ordinal])[0],feature_type))
    no_match_vector=lb.transform([-1])[0]
    # regex for feature -> feature vector, no-Match_vector = the feature vector for no matching regex
    return (regex_features, no_match_vector)

def phrase_match(phrase, sentence):
    import re
    return True if re.search(phrase,sentence) else False # match objects in python evaluate to True


# deals with wierd pandas nulls
def clean(x):
    return "" if pd.isnull(x) else str(x)


# generates different corpus files only containing sentences which contain a search term relevant to the study
# f_note_id_to_writer : a function which takes a study_id and a note_id and returns a writer for writing the sentence. implement this to obtain different file splits
# f_get_sentence_label : function which returns the label for the setence 0/1, dictionary: studyd-> list of regex phrases
# filf_filter an optional filter function.  return True to avoid writing this sentence to the output writer.  signature  = StudyId,PatientID,CaseLabel,Sublabel,DiagnosisDate,NoteID,SourceNoteRecordedDate,Sentence_orig
def filter_negation_setences(f_meta_data_to_writer, study_phrases, f_get_sentence_label, negation_sentences_file = 'negation_detection_sentences.txt', folder = data_raw_folder, f_filter=None):
    import collections
    df=pd.read_csv(folder + '/' + negation_sentences_file, sep=',',header=0,encoding="utf-8",escapechar="\\",quotechar="\"")
    sentences = 0
    sentences_total=0
    sentences_filtered_out=0
    counts={} #study to tuple : (pos,neg)

    for index, row in df.iterrows():
        StudyId,PatientID,CaseLabel,Sublabel,DiagnosisDate,NoteID,SourceNoteRecordedDate,Sentence_orig = row
        sentences_total = sentences_total + 1

         # check to see if we should filter out the example
        if f_filter is not None:
            filter_out = f_filter(StudyId,PatientID,CaseLabel,Sublabel,DiagnosisDate,NoteID,SourceNoteRecordedDate,Sentence_orig)
            if filter_out:
                sentences_filtered_out = sentences_filtered_out + 1
                continue

        # retokenise sentence to split words containing / or \
        tokens = tokenize(Sentence_orig)
        sentence = ' '.join(tokens)
        if str(StudyId) not in study_phrases:
            print('couldn''t find study in study_phrases!')
            print(str(StudyId))
            print(Sentence_orig)
            continue

        phrases = study_phrases[str(StudyId)]

        if any(phrase_match(phrase, sentence) for phrase in phrases):
            sentences = sentences + 1
            if sentences % 20000 == 0:
                print ('phrase matched on %s/%s' %(sentences,sentences_total))
            sentence_label = f_get_sentence_label(StudyId,CaseLabel,Sublabel,DiagnosisDate,SourceNoteRecordedDate)
            sentence = sentence.replace("\"","\"\"")
            sublabel = clean(Sublabel)
            dd=clean(DiagnosisDate)
            nd=clean(SourceNoteRecordedDate)
            line = "\"%s\",%s,%s,\"%s\",\"%s\",%s,%s,%s,\"%s\"\n" % (StudyId,PatientID,NoteID,CaseLabel,sublabel,sentence_label,dd,nd,sentence)

            # get the correct writer based on the noteid
            writer = f_meta_data_to_writer(StudyId, NoteID)
            writer.write(line)
            
            # update counts
            (pos_count, neg_count) = counts.get(StudyId,(0,0))
            if sentence_label > 0:
                pos_count = pos_count + 1
            else:
                neg_count = neg_count + 1
            counts[StudyId]=(pos_count, neg_count)

    print ('phrase matched on %s/%s' %(sentences,sentences_total))
    print ('sentences_filtered_out = %s' % sentences_filtered_out)
    return counts

# filter the raw sentences from notes2corpus.exe, hold out any notes from the 
# gold standard so they dont go into the silver standard
def save_gold_and_silver_sentences():
    # get the gold standard noteids
    gold_standard_note_ids = set()
    df=pd.read_csv(data_raw_folder + '/' + 'gold standard likely is 0.csv', sep=',',header=0)
    for index, row in df.iterrows():
        StudyId,PatientID,NoteId,CaseLabel,Sublabel,SentenceLabel,DiagnosisDate,SourceNoteRecordedDate,Sentence = row
        gold_standard_note_ids.add(NoteId)

    with open(data_processed_folder+ '/negation_detection_sentences_silver_standard.txt' ,'w') as writer_silver:
        with open(data_processed_folder+ '/negation_detection_sentences_gold_standard.txt' ,'w') as writer_gold:
            # if note_id was used in gold standard, then gold, else silver. This rule prevents any training examples
            # which are part of the gold evaluation from being used in training.  this rule will stop notes which are
            # in more than one study from being used in training if one of the studies was part of the original 6

            def silver_gold_split(study_id, note_id):
                if note_id in gold_standard_note_ids:
                    return writer_gold
                else:
                    return writer_silver
            filter_negation_setences(silver_gold_split,get_study_phrases(),get_sentence_label_primary_dataset)
            writer_gold.flush()
            writer_gold.close()
        writer_silver.flush()
        writer_silver.close()

# filters the x million referral dataset sentences by dropping sentences which dont have a study phrase
# also filters out the gold set
def create_referral_dataset_setences_from_output_of_notes2Corpus(negation_sentences_file='referral_negation_detection_sentences_experiment_6.txt'):
    # need to make sure we are filtering out the gold set as well
    gold_set_df=pd.read_csv(data_processed_folder + '/' + 'gold referral 2 is 0.csv', sep=',',header=0,encoding="utf-8",escapechar="\\",quotechar="\"")

    gold_set = set()
    for index, row in gold_set_df.iterrows():
        StudyId,PatientID,Noteid,CaseLabel,Sublabel,SentenceLabel,DiagnosisDate,SourceNoteRecordedDate,Sentence_orig = row
        Sentence_orig = Sentence_orig.replace('__* ','').replace(' *__','')
        gold_key='_'.join(map(lambda x:str(x),[Sublabel,SourceNoteRecordedDate,Sentence_orig]))# same patient, same note time, same sentence = match between training and gold.  Exclude from training.
        gold_set.add(gold_key)

    def filter_out_gold(StudyId,PatientID,CaseLabel,Sublabel,DiagnosisDate,NoteID,SourceNoteRecordedDate,Sentence_orig):
        gold_key='_'.join(map(lambda x:str(x),[Sublabel,SourceNoteRecordedDate,Sentence_orig]))
        if gold_key in gold_set:
            print('found a member of the gold set sentence in the source data')
            print(gold_key)
            return True
        else:
            return False

    # produces a label 1/0 for a sentence based on the study & note meta data : referral dataset
    def get_sentence_label_referral_dataset(StudyId,Label,Sublabel,DiagnosisDate,SourceNoteRecordedDate):
        if str(Label)=="0": return 0.0
        elif str(Label)=="1": return 1.0
        else:
            print(Label)
            print(StudyId)
            raise ValueError('dunno what this label is?! was expecting 0/1')

    study_phrases = get_referral_study_phrases()

    with open(data_processed_folder+ '/referral_dataset_sentences_silver_standard_experiment_6.txt' ,'w', encoding='utf-8') as writer:
        writer.write("StudyId,PatientID,NoteId,CaseLabel,Sublabel,SentenceLabel,DiagnosisDate,SourceNoteRecordedDate,Sentence\n")
        def study_to_writer(study_id, note_id):
            return writer # just one writer, we aren't partitioning

        counts=filter_negation_setences(study_to_writer,study_phrases, get_sentence_label_referral_dataset, negation_sentences_file, f_filter=filter_out_gold)
        writer.flush()
        writer.close()
        return counts

# produces 5 fold split of sentences.  writer_map  = dict [int 0-9] -> writer
def noteid_to_5_fold(writer_map, note_id):
    random = int(str(hash(str(note_id) + randomisation_salt))[-1:])
    return writer_map[random]

# a function to create a 5-fold split of negation sentences
# this works on the silver standard, not the raw sentences.  The silver standard has already been filtered so all sentences are relevent 
def save_5_fold_split():
    from functools import partial

    writer_map = dict()
    for i in range(0,5):
        writer = open(data_processed_folder+ '/negation_detection_sentences_5_fold_%s.txt' % i ,'w')
        writer_map[i] = writer
        writer_map[i+5] = writer

    df=pd.read_csv(data_processed_folder + '/' + 'negation_detection_sentences_silver_standard.txt', sep=',',header=1)

    for index, row in df.iterrows():
        StudyId,PatientId,NoteId,CaseLabel,Sublabel,SentenceLabel,DiagnosisDate,SourceNoteRecordedDate,Sentence = row
        sentence = Sentence.replace("\"","\"\"")
        sublabel = clean(Sublabel)
        dd=clean(DiagnosisDate)
        nd=clean(SourceNoteRecordedDate)
        line = "\"%s\",%s,%s,\"%s\",\"%s\",%s,%s,%s,\"%s\"\n" % (StudyId,PatientId,NoteId,CaseLabel,sublabel,SentenceLabel,dd,nd,sentence)

        # get the correct writer based on the noteid
        random = int(str(hash(str(NoteId) + randomisation_salt))[-1:])
        writer = noteid_to_5_fold(writer_map, NoteId)
        writer.write(line)


    for i in range(0,5):
        writer = writer_map[i]
        writer.flush()
        writer.close()

# this surrounds the disease reference in a sentence with __* *__ so annotators know what disease they are labelling
def markup_gold_standard(study_phrases,file='negation_detection_sentences_gold_standard.txt'):
    import re

    df=pd.read_csv(data_processed_folder + '/' + file, sep=',',header=0)
    #study_phrases=get_study_phrases()

    with open(data_processed_folder+ '/negation_detection_sentences_gold_standard_phrases_marked.csv',r'w') as writer_gold:
        writer_gold.write(header)
        for index, row in df.iterrows():
            StudyId,PatientId,NoteId,CaseLabel,Sublabel,SentenceLabel,DiagnosisDate,SourceNoteRecordedDate,Sentence = row
            sentence = Sentence.replace("\"","\"\"")
            sublabel = clean(Sublabel)
            dd=clean(DiagnosisDate)
            nd=clean(SourceNoteRecordedDate)

            study_phrase_set = study_phrases[str(StudyId).lower()]

            # surround the first match with the highlightinh token. this means annotator knows what they are looking at
            for regex in study_phrase_set:
                match = re.search(regex,sentence)
                if not match: continue

                token = sentence[match.span()[0]:match.span()[1]]
                token = "__* %s *__" % token
                sentence = sentence[:match.span()[0]] + token + sentence[match.span()[1]:]
                break

            line = "\"%s\",%s,%s,\"%s\",\"%s\",%s,%s,%s,\"%s\"\n" % (StudyId,PatientId,NoteId,CaseLabel,sublabel,SentenceLabel,dd,nd,sentence)
            writer_gold.write(line)
        writer_gold.flush()
        writer_gold.close()

def save_np_models(x,y, file_name_suffix='train'):
    np.save('x_%s' % file_name_suffix,x)
    np.save('y_%s' % file_name_suffix,y)

#def load_true_gold_standard(word2vec,max_sentence_length):


# samples gold_standard_sample_rate % of notes for manual classificaiton to a true gold standard 
def save_random_sample_for_true_gold_standard(gold_standard_sample_rate = 0.04):
    with open(data_processed_folder+ '/negation_detection_sentences_silver_standard.txt\b',r'\bw') as writer_silver:
        with open(data_processed_folder+ '/negation_detection_sentences_gold_standard.txt\b',r'\bw') as writer_gold:
            writer_silver.write(header)
            writer_gold.write(header)

            def meta_data_to_silver_or_gold(study_id, note_id):
                # use the last 2 numbers of the random hash to assign to gold or silver
                random = int(str(hash(str(note_id) + randomisation_salt))[-2:])
                if random < (gold_standard_sample_rate * 100):
                    return writer_gold
                else:
                    return writer_silver
    
            filter_negation_setences(meta_data_to_silver_or_gold,get_study_phrases(),get_sentence_label_primary_dataset)
            writer_gold.flush()
            writer_silver.flush()


# this is the original train, dev, test split
# 80 % of notes assigned to train, 10% to dev 10% to test
def noteid_to_writer_train_dev_test(writer_train, writer_dev, writer_test, note_id):
    from random import randint
    random = randint(0, 9)
    if random <=7:
        return writer_train
    elif random == 9:
        return writer_test
    else:
        return writer_dev

# a function to create a train, dev, test split of negation sentences.  
def save_train_dev_test_split(file_name='referral_negation_detection_sentences_experiment_8.txt'):
    from functools import partial

    with open(data_processed_folder+ '/negation_detection_sentences_train.txt','w') as writer_train:
        with open(data_processed_folder+ '/negation_detection_sentences_dev.txt','w') as writer_dev:
            with open(data_processed_folder+ '/negation_detection_sentences_test.txt','w') as writer_test:
                writer_train.write(header)
                writer_dev.write(header)
                writer_test.write(header)
                g = partial(noteid_to_writer_train_dev_test, writer_train, writer_dev, writer_test)

                df=pd.read_csv(data_processed_folder + '/' + file_name, sep=',',header=0)
                for index, row in df.iterrows():
                    StudyID,PatientID,NoteID,CaseLabel,Sublabel,SentenceLabel,DiagnosisDate,SourceNoteRecordedDate,Sentence = row
                    split_writer = g(NoteID)
                    sentence = Sentence.replace("\"","\"\"")
                    sublabel = clean(Sublabel)
                    dd=clean(DiagnosisDate)
                    nd=clean(SourceNoteRecordedDate)
                    line = "\"%s\",%s,%s,\"%s\",\"%s\",%s,%s,%s,\"%s\"\n" % (StudyID,PatientID,NoteID,CaseLabel,sublabel,SentenceLabel,dd,nd,sentence)
                    split_writer.write(line) 
                writer_test.flush()
                writer_test.close()
            writer_dev.flush()
            writer_dev.close()
        writer_train.flush()
        writer_train.close()

def auc(y_test,y_pred):
    import sklearn.metrics
    return sklearn.metrics.roc_auc_score(y_test,y_pred)

def roc(y_test,y_pred, title='Receiver Operating Characteristic'):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.xlabel('1-Specificity (False Positive Rate)')
    plt.show()

# draw a ROC with a baseline on the same chart
def roc_baseline(y_test, y_pred_base, y_pred_cnn, title='Receiver Operating Characteristic'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_base)
    fpr_model, tpr_model, _ = roc_curve(y_test, y_pred_cnn)

    roc_auc_base = auc(fpr_base, tpr_base)
    roc_auc_model = auc(fpr_model, tpr_model)

    plt.gcf().clear()
    plt.title(title)
    plt.plot(fpr_base, tpr_base, 'b', label = 'UTH-CCB AUC = %0.2f' % roc_auc_base, linestyle=':')
    plt.plot(fpr_model, tpr_model, 'b', label = 'CNN AUC = %0.2f' % roc_auc_model)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.xlabel('1-Specificity (False Positive Rate)')
    plt.show()
    return plt

