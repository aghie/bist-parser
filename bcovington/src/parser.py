from argparse import ArgumentParser
import utils
import covington
import os
import pickle
import time
import tempfile
import yaml
import codecs
import sys
import warnings
"""
Main file
"""



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Path to the input file",default=None)
    parser.add_argument("--input_type", dest="input_type",help="Style of the input file [raw|conllu] (only use with --predict)")
    parser.add_argument("--pipe", dest="pipe",default="UDpipe",help="Framework used to do the pipeline. Only \"UDpipe\" supported (only use with --predict)")
    parser.add_argument("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="../data/PTB_SD_3_3_0/train.conll")
    parser.add_argument("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="../data/PTB_SD_3_3_0/dev.conll")
    parser.add_argument("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="../data/PTB_SD_3_3_0/test.conll")
    parser.add_argument("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_argument("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_argument("--extrn_cpos", dest="cpos_external_embedding",help="CPoStag external embeddings", metavar="FILE")
    parser.add_argument("--extrn_pos", dest="pos_external_embedding", help= "PoStag external embeddings", metavar="FILE")
    parser.add_argument("--extrn_feats", dest="feats_external_embedding", help="Feats external embeddings", metavar="FILE")
    parser.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", default="bcovington.model")
    parser.add_argument("--wembedding", type=int, dest="wembedding_dims", default=100)
    parser.add_argument("--pembedding", type=int, dest="pembedding_dims", default=25)
    parser.add_argument("--rembedding", type=int, dest="rembedding_dims", default=25)
    parser.add_argument("--epochs", type=int, dest="epochs", default=30)
    parser.add_argument("--hidden", type=int, dest="hidden_units", default=100)
    parser.add_argument("--hidden2", type=int, dest="hidden2_units", default=0)
    parser.add_argument("--kb", type=int, dest="window_b", default=1)
    parser.add_argument("--k1", type=int, dest="window_l1", default=3)
    parser.add_argument("--k2r", type=int, dest="window_l2r", default = 1)
    parser.add_argument("--k2l", type=int, dest="window_l2l", default = 1)  
    parser.add_argument("--lr", type=float, dest="learning_rate", default=0.1)
    parser.add_argument("--outdir", type=str, dest="output", default="results")
    parser.add_argument("--activation", type=str, dest="activation", default="tanh")
    parser.add_argument("--optimizer",type=str, dest="optimizer", default="adam")
    parser.add_argument("--lstmlayers", type=int, dest="lstm_layers", default=2)
    parser.add_argument("--lstmdims", type=int, dest="lstm_dims", default=125)
    parser.add_argument("--dynet-seed", type=int, dest="seed", default=7)
    parser.add_argument("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_argument("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_argument("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_argument("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_argument("--userlmost", action="store_true", dest="rlFlag", default=False)
    parser.add_argument("--userl", action="store_true", dest="rlMostFlag", default=False)
    parser.add_argument("--dynet-mem", type=int, dest="cnn_mem", default=512)
    parser.add_argument("--conll2017", action="store_true",dest="conll2017", default=False)
    parser.add_argument("--predict", action="store_true", dest="predictFlag", default=False)

    
   # parser.add_argument("--conf", metavar="FILE", dest="conf",required=True)

    args = parser.parse_args()

    if not args.predictFlag:

        if not os.path.exists(args.output):
            os.mkdir(args.output)
            
       # config = yaml.safe_load(open(args.conf))
    
        print "Training..."
        if not (args.rlFlag or args.rlMostFlag or args.headFlag):
            print 'You must use either --userlmost or --userl or --usehead (you can use multiple)'
            sys.exit()
        
        path_tmp_file_oov = None
        
        print 'Preparing vocab'
        words, w2i, lemmas, l2i, cpos, pos, feats, rels = utils.vocab(args.conll_train)
    

        with open(os.path.join(args.output, args.params), 'w') as paramsfp:
            pickle.dump((words, w2i, lemmas, l2i, cpos, pos, feats, rels, args), paramsfp)
        print 'Finished collecting vocab'
    
        print 'Initializing blstm covington:'
        parser = covington.CovingtonBILSTM(words, lemmas, cpos, pos, feats, rels, w2i, l2i, args, 
                                           path_tmp_file_oov)
     
    
        if path_tmp_file_oov is not None:
            os.unlink(path_tmp_file_oov)
        
        if args.conll2017:
            with codecs.open(args.conll_dev) as f_conll_dev:
                lookup_conll_data = utils.lookup_conll_extra_data(f_conll_dev)            
                    
        
        
        for epoch in xrange(args.epochs):
            print 'Starting epoch', epoch
            parser.Train(args.conll_train)
            devpath = os.path.join(args.output, 'dev_epoch_' + str(epoch+1) + '.conll')
            utils.write_conll(devpath, parser.Predict(args.conll_dev))
            
            if args.conll2017:
                utils.dump_lookup_extra_into_conll(devpath, lookup_conll_data)
                utils.transform_to_single_root(devpath)
            
            
            print 'Executing conll17_eval'
            

            if not args.conll2017:
                os.system('perl src/utils/eval.pl -g ' + args.conll_dev + ' -s ' + devpath  + ' > ' + devpath + '.txt')
            else:
                os.system('python src/utils/conll17_ud_eval.py -v -w src/utils/weights.clas ' + args.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
        



            parser.Save(os.path.join(args.output, args.model))
            
    else:
        
        #TEST PHASE
        with codecs.open(args.params, 'r') as paramsfp:
            aux = pickle.load(paramsfp)
            words, w2i, lemmas, l2i, cpos , pos, feats, rels, stored_opt = aux                 
                        
        
        stored_opt.external_embedding = args.external_embedding
        stored_opt.pos_external_embedding = args.pos_external_embedding
        stored_opt.feats_external_embedding = args.feats_external_embedding
    
        print "Running model with this configuration", stored_opt
    
        parser = covington.CovingtonBILSTM(words, lemmas, cpos, pos, feats, rels, w2i, l2i, stored_opt,
                                                                                       None)
                            
        parser.Load(args.model)
                            
        conllu = (os.path.splitext(args.conll_test.lower())[1] == '.conllu')
        tespath = os.path.join(args.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')
        
        
        if args.conll2017:
            with codecs.open(args.conll_test) as f_conll_test:
                lookup_conll_data = utils.lookup_conll_extra_data(f_conll_test)            
                    
        
        
        ts = time.time()
        pred = list(parser.Predict(args.conll_test))
        te = time.time()
        utils.write_conll(tespath, pred)
        
        
        if args.conll2017:
            utils.dump_lookup_extra_into_conll(tespath, lookup_conll_data)
            utils.transform_to_single_root(tespath)
            

        if not args.conll2017:
            os.system('perl src/utils/eval.pl -g ' + args.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.txt')
        else:
            os.system('python src/utils/conll17_ud_eval.py -v -w src/utils/weights.clas ' + args.conll_test + ' ' + tespath + ' > ' + tespath + '.txt')
        


                            
                            

    
