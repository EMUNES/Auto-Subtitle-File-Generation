// Copyright 2019-2021 Alpha Cephei Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


//
// For details of possible model layout see doc/models.md section model-structure

#include "model.h"

#include <sys/stat.h>
#include <fst/fst.h>
#include <fst/register.h>
#include <fst/matcher-fst.h>
#include <fst/extensions/ngram/ngram-fst.h>

namespace fst {

static FstRegisterer<StdOLabelLookAheadFst> OLabelLookAheadFst_StdArc_registerer;
static FstRegisterer<NGramFst<StdArc>> NGramFst_StdArc_registerer;

}  // namespace fst

#ifdef __ANDROID__
#include <android/log.h>
static void KaldiLogHandler(const LogMessageEnvelope &env, const char *message)
{
  int priority;
  if (env.severity > GetVerboseLevel())
      return;

  if (env.severity > LogMessageEnvelope::kInfo) {
    priority = ANDROID_LOG_VERBOSE;
  } else {
    switch (env.severity) {
    case LogMessageEnvelope::kInfo:
      priority = ANDROID_LOG_INFO;
      break;
    case LogMessageEnvelope::kWarning:
      priority = ANDROID_LOG_WARN;
      break;
    case LogMessageEnvelope::kAssertFailed:
      priority = ANDROID_LOG_FATAL;
      break;
    case LogMessageEnvelope::kError:
    default: // If not the ERROR, it still an error!
      priority = ANDROID_LOG_ERROR;
      break;
    }
  }

  std::stringstream full_message;
  full_message << env.func << "():" << env.file << ':'
               << env.line << ") " << message;

  __android_log_print(priority, "VoskAPI", "%s", full_message.str().c_str());
}
#else
static void KaldiLogHandler(const LogMessageEnvelope &env, const char *message)
{
  if (env.severity > GetVerboseLevel())
      return;

  // Modified default Kaldi logging so we can disable LOG messages.
  std::stringstream full_message;
  if (env.severity > LogMessageEnvelope::kInfo) {
    full_message << "VLOG[" << env.severity << "] (";
  } else {
    switch (env.severity) {
    case LogMessageEnvelope::kInfo:
      full_message << "LOG (";
      break;
    case LogMessageEnvelope::kWarning:
      full_message << "WARNING (";
      break;
    case LogMessageEnvelope::kAssertFailed:
      full_message << "ASSERTION_FAILED (";
      break;
    case LogMessageEnvelope::kError:
    default: // If not the ERROR, it still an error!
      full_message << "ERROR (";
      break;
    }
  }
  // Add other info from the envelope and the message text.
  full_message << "VoskAPI" << ':'
               << env.func << "():" << env.file << ':'
               << env.line << ") " << message;

  // Print the complete message to stderr.
  full_message << "\n";
  std::cerr << full_message.str();
}
#endif

Model::Model(const char *model_path) : model_path_str_(model_path) {

    SetLogHandler(KaldiLogHandler);

    struct stat buffer;
    string am_path = model_path_str_ + "/am/final.mdl";
    if (stat(am_path.c_str(), &buffer) == 0) {
         ConfigureV2();
    } else {
         ConfigureV1();
    }

    ReadDataFiles();

    ref_cnt_ = 1;
}

// Old model layout without model configuration file

void Model::ConfigureV1()
{
    const char *extra_args[] = {
        "--max-active=7000",
        "--beam=13.0",
        "--lattice-beam=6.0",
        "--acoustic-scale=1.0",

        "--frame-subsampling-factor=3",

        "--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10",
        "--endpoint.rule2.min-trailing-silence=0.5",
        "--endpoint.rule3.min-trailing-silence=1.0",
        "--endpoint.rule4.min-trailing-silence=2.0",

        "--print-args=false",
    };

    kaldi::ParseOptions po("");
    nnet3_decoding_config_.Register(&po);
    endpoint_config_.Register(&po);
    decodable_opts_.Register(&po);

    vector<const char*> args;
    args.push_back("vosk");
    args.insert(args.end(), extra_args, extra_args + sizeof(extra_args) / sizeof(extra_args[0]));
    po.Read(args.size(), args.data());

    nnet3_rxfilename_ = model_path_str_ + "/final.mdl";
    hclg_fst_rxfilename_ = model_path_str_ + "/HCLG.fst";
    hcl_fst_rxfilename_ = model_path_str_ + "/HCLr.fst";
    g_fst_rxfilename_ = model_path_str_ + "/Gr.fst";
    disambig_rxfilename_ = model_path_str_ + "/disambig_tid.int";
    word_syms_rxfilename_ = model_path_str_ + "/words.txt";
    winfo_rxfilename_ = model_path_str_ + "/word_boundary.int";
    carpa_rxfilename_ = model_path_str_ + "/rescore/G.carpa";
    std_fst_rxfilename_ = model_path_str_ + "/rescore/G.fst";
    final_ie_rxfilename_ = model_path_str_ + "/ivector/final.ie";
    mfcc_conf_rxfilename_ = model_path_str_ + "/mfcc.conf";
    global_cmvn_stats_rxfilename_ = model_path_str_ + "/global_cmvn.stats";
    pitch_conf_rxfilename_ = model_path_str_ + "/pitch.conf";
}

void Model::ConfigureV2()
{
    kaldi::ParseOptions po("something");
    nnet3_decoding_config_.Register(&po);
    endpoint_config_.Register(&po);
    decodable_opts_.Register(&po);
    po.ReadConfigFile(model_path_str_ + "/conf/model.conf");


    nnet3_rxfilename_ = model_path_str_ + "/am/final.mdl";
    hclg_fst_rxfilename_ = model_path_str_ + "/graph/HCLG.fst";
    hcl_fst_rxfilename_ = model_path_str_ + "/graph/HCLr.fst";
    g_fst_rxfilename_ = model_path_str_ + "/graph/Gr.fst";
    disambig_rxfilename_ = model_path_str_ + "/graph/disambig_tid.int";
    word_syms_rxfilename_ = model_path_str_ + "/graph/words.txt";
    winfo_rxfilename_ = model_path_str_ + "/graph/phones/word_boundary.int";
    carpa_rxfilename_ = model_path_str_ + "/rescore/G.carpa";
    std_fst_rxfilename_ = model_path_str_ + "/rescore/G.fst";
    final_ie_rxfilename_ = model_path_str_ + "/ivector/final.ie";
    mfcc_conf_rxfilename_ = model_path_str_ + "/conf/mfcc.conf";
    global_cmvn_stats_rxfilename_ = model_path_str_ + "/am/global_cmvn.stats";
    pitch_conf_rxfilename_ = model_path_str_ + "/conf/pitch.conf";
}

void Model::ReadDataFiles()
{
    struct stat buffer;

    KALDI_LOG << "Decoding params beam=" << nnet3_decoding_config_.beam <<
         " max-active=" << nnet3_decoding_config_.max_active <<
         " lattice-beam=" << nnet3_decoding_config_.lattice_beam;
    KALDI_LOG << "Silence phones " << endpoint_config_.silence_phones;

    feature_info_.feature_type = "mfcc";
    ReadConfigFromFile(mfcc_conf_rxfilename_, &feature_info_.mfcc_opts);
    feature_info_.mfcc_opts.frame_opts.allow_downsample = true; // It is safe to downsample

    feature_info_.silence_weighting_config.silence_weight = 1e-3;
    feature_info_.silence_weighting_config.silence_phones_str = endpoint_config_.silence_phones;

    trans_model_ = new kaldi::TransitionModel();
    nnet_ = new kaldi::nnet3::AmNnetSimple();
    {
        bool binary;
        kaldi::Input ki(nnet3_rxfilename_, &binary);
        trans_model_->Read(ki.Stream(), binary);
        nnet_->Read(ki.Stream(), binary);
        SetBatchnormTestMode(true, &(nnet_->GetNnet()));
        SetDropoutTestMode(true, &(nnet_->GetNnet()));
        nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(nnet_->GetNnet()));
    }
    decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts_,
                                                               nnet_);

    if (stat(final_ie_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Loading i-vector extractor from " << final_ie_rxfilename_;

        OnlineIvectorExtractionConfig ivector_extraction_opts;
        ivector_extraction_opts.splice_config_rxfilename = model_path_str_ + "/ivector/splice.conf";
        ivector_extraction_opts.cmvn_config_rxfilename = model_path_str_ + "/ivector/online_cmvn.conf";
        ivector_extraction_opts.lda_mat_rxfilename = model_path_str_ + "/ivector/final.mat";
        ivector_extraction_opts.global_cmvn_stats_rxfilename = model_path_str_ + "/ivector/global_cmvn.stats";
        ivector_extraction_opts.diag_ubm_rxfilename = model_path_str_ + "/ivector/final.dubm";
        ivector_extraction_opts.ivector_extractor_rxfilename = model_path_str_ + "/ivector/final.ie";
        ivector_extraction_opts.max_count = 100;

        feature_info_.use_ivectors = true;
        feature_info_.ivector_extractor_info.Init(ivector_extraction_opts);
    } else {
        feature_info_.use_ivectors = false;
    }

    if (stat(global_cmvn_stats_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Reading CMVN stats from " << global_cmvn_stats_rxfilename_;
        feature_info_.use_cmvn = true;
        ReadKaldiObject(global_cmvn_stats_rxfilename_, &feature_info_.global_cmvn_stats);
    }

    if (stat(pitch_conf_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Using pitch in feature pipeline";
        feature_info_.add_pitch = true;
        ReadConfigFromFile(pitch_conf_rxfilename_, &feature_info_.pitch_opts);
    }

    if (stat(hclg_fst_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Loading HCLG from " << hclg_fst_rxfilename_;
        hclg_fst_ = fst::ReadFstKaldiGeneric(hclg_fst_rxfilename_);
        hcl_fst_ = NULL;
        g_fst_ = NULL;
    } else {
        KALDI_LOG << "Loading HCL and G from " << hcl_fst_rxfilename_ << " " << g_fst_rxfilename_;
        hclg_fst_ = NULL;
        hcl_fst_ = fst::StdFst::Read(hcl_fst_rxfilename_);
        g_fst_ = fst::StdFst::Read(g_fst_rxfilename_);
        ReadIntegerVectorSimple(disambig_rxfilename_, &disambig_);
    }

    word_syms_ = NULL;
    word_syms_loaded_ = false;
    if (hclg_fst_ && hclg_fst_->OutputSymbols()) {
        word_syms_ = hclg_fst_->OutputSymbols();
    } else if (g_fst_ && g_fst_->OutputSymbols()) {
        word_syms_ = g_fst_->OutputSymbols();
    }
    if (!word_syms_) {
        KALDI_LOG << "Loading words from " << word_syms_rxfilename_;
        if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename_)))
            KALDI_ERR << "Could not read symbol table from file "
                      << word_syms_rxfilename_;
        word_syms_loaded_ = word_syms_;
    }
    KALDI_ASSERT(word_syms_);

    if (stat(winfo_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Loading winfo " << winfo_rxfilename_;
        kaldi::WordBoundaryInfoNewOpts opts;
        winfo_ = new kaldi::WordBoundaryInfo(opts, winfo_rxfilename_);
    } else {
        winfo_ = NULL;
    }

    std_lm_fst_ = NULL;
    if (stat(carpa_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Loading CARPA model from " << carpa_rxfilename_;
        std_lm_fst_ = fst::ReadFstKaldi(std_fst_rxfilename_);
        fst::Project(std_lm_fst_, fst::PROJECT_OUTPUT);
        if (std_lm_fst_->Properties(fst::kILabelSorted, true) == 0) {
            fst::ILabelCompare<fst::StdArc> ilabel_comp;
            fst::ArcSort(std_lm_fst_, ilabel_comp);
        }
        ReadKaldiObject(carpa_rxfilename_, &const_arpa_);
    }
}

void Model::Ref() 
{
    ref_cnt_++;
}

void Model::Unref() 
{
    ref_cnt_--;
    if (ref_cnt_ == 0) {
        delete this;
    }
}

int Model::FindWord(const char *word)
{
    if (!word_syms_)
        return -1;

    return word_syms_->Find(word);
}

Model::~Model() {
    delete decodable_info_;
    delete trans_model_;
    delete nnet_;
    if (word_syms_loaded_)
        delete word_syms_;
    delete winfo_;
    delete hclg_fst_;
    delete hcl_fst_;
    delete g_fst_;
    delete std_lm_fst_;
}
