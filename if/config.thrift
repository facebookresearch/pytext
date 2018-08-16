namespace py pytext.config

// This interface defines the parameters schema of the different PyText models
// A PyText config consists of 6 main sections:
//    1- Data parameters: Defines the data paths and any data preprocessing
//                        parameters
//    2- Features parameters: Defines what features to use in the model and
//                            their embeddings dimension
//    3- Training parameters: Defines the basic training runtime parameters
//                            and the optimizer config
//    4- Model specific parameters: Defines the parameters that are specific to
//                                  the different models implemented in PyText
//    5- Output paths: Defines the different paths where to save the different
//                      outputs: model snapshot, debug_file, learnt embeddings
//    6- Other Miscellaneous parameters: use cuda or not, run test only, etc.

struct DataParams {
  // Base directory that contains the data splits
  1: string base_path = "./",
  2: string train_filename = "train.tsv",
  3: string eval_filename = "eval.tsv",
  4: string test_filename = "test.tsv",
  // Number of parallel workers to preprocess the data
  5: i32 preprocess_workers = 32,
  // Shuffle training data
  6: bool shuffle = true,
  // Transform sequence labels to BIO format
  7: bool use_bio_labels = false,
  // Deprecated
  // 8: string model_metadata = "";
  // Used for multilingual models
  9: bool include_language = false,
  10: map<string,string> i18n_tokenizer_configs,
  11: bool include_alignment = false,
  12: i32 max_train_num = -1,
  13: i32 max_dev_num = -1,
  14: i32 max_test_num = -1,
}

enum PoolingType {
  MEAN = 0,
  MAX = 1,
}

struct FeatureParams {
  // Word embeddings dimension
  1: i32 embed_dim = 100,
  // Character embeddings dimension
  2: i32 char_embed_dim = 0,
  // Dictionary feature embeddings dimension
  // Set to zero to disable dictionary features
  4: i32 dict_feat_dim = 0,
  // Capitalization feature embeddings dimension
  // Set to zero to disable Capitalization features
  5: i32 cap_feat_dim = 0,
  // Pretrained embeddings package
  6: string pretrained_embeds_pkg,
  // Dict feats pooling type
  8: PoolingType dict_feat_pooling = MEAN,
  // Path to pretrained embeddings file.
  10: string pretrained_embeds_path,
  // CNN parameters for charCNN embeddings
  11: CNNParams char_cnn,
  // User query sentence embedding dimension
  // Set to zero to disable query embedding
  12: i32 query_emb_dim = 0,
  // Option to freeze word embeddings
  13: bool freeze_word_embeds = false,
  // Embedding initialization strategy
  15: string embed_intialization_strategy = "random",
}

enum OptimizerType {
  // ADAM optimizer implemented here: https://fburl.com/7nerg6v4
  ADAM = 0,
  // SGD optimizer implemented here: https://fburl.com/mlj4q7ik
  SGD = 1,
}

struct OptimizerParams {
  1: OptimizerType type = ADAM,
  // Learning rate
  2: float lr = 0.001,
  3: float weight_decay = 0.00001,
  4: float momentum = 0.0,
  5: bool use_sparse_grad_for_embed = false,
  6: string embedding_param_prefix = '',
}

enum ClassifierLoss {
  CROSS_ENTROPY = 0,
  BINARY_CROSS_ENTROPY = 1,
}

enum TaggerLoss {
  CROSS_ENTROPY = 0,
  CRF = 1,
}

enum LanguageModelLoss{
  CROSS_ENTROPY = 0
}

struct LossParams {
  1: ClassifierLoss classifier_loss = ClassifierLoss.CROSS_ENTROPY,
  2: TaggerLoss tagger_loss = TaggerLoss.CROSS_ENTROPY,
  3: LanguageModelLoss language_model_loss = LanguageModelLoss.CROSS_ENTROPY,
}

struct TrainingParams {
  // Manual random seed
  1: i32 random_seed = 0,
  // Training batch_size
  2: i32 batch_size = 128,
  // Eval batch_size
  3: i32 eval_batch_size = 128,
  // Training epochs
  4: i32 epochs = 10,
  // Stop after how many epochs when the eval metric is not improving
  5: i32 early_stop_after = 0,
  // Print the training metrics every log_interval epochs
  6: i32 log_interval = 1,
  // Evaluate the model every eval_interval epochs
  7: i32 eval_interval = 1,
  // Number of parallel workers when the model is using hogwild training
  8: i32 num_workers = 1,
  // when set, it will enable using visodom to visualize the model
  // internals (https://fburl.com/ts638hk7)
  9: string visdom_exp_name,
  10: OptimizerParams optimizer_params,
  11: LossParams loss_params,
}

union DocModel {
  1: DocBLSTM docblstm,
  2: DocNN docnn,
}

union WordModel {
  1: WordBLSTM wordblstm,
  2: WordCNN wordcnn,
}

# until we find a better solution, the enum ids need to be coupled manually
# with the Model union ids
union JointModel {
  4: JointBLSTM jointblstm,
  14: JointCNN jointcnn,
}

union PairModel {
  1: PairNN pairnn,
}

struct BaggingDocEnsemble {
  1: float sample_rate = 1.0,
  2: list<DocModel> models,
}

struct BaggingJointEnsemble {
  1: float sample_rate = 1.0,
  2: list<JointModel> models,
  3: bool use_crf = false,
}

struct OutputPaths {
  // A file to store the output of the model when running on the test data
  1: string test_out_path = "/tmp/test_out.txt",
  // Where to save the trained model
  2: string save_snapshot_path = "/tmp/model.pt",
  // A file to store model debug information
  3: string debug_path = "/tmp/model.debug",
  // When the export flag is on, the exported model will be stored here
  4: string fbl_pred_path = "/tmp/model.fbl.predictor",
  // Deprecated
  // 5: string model_metadata = "/tmp/model_metadata.pkl",
}

struct DecodingParams {
  // beam capacity when using beam search
  1: i32 beam_k = 0,
  // softmax temperature to use while decoding
  // default value of 0 indicates that next token will always be chosen by
  // argmax rather than sampling
  2: float temperature = 0.0,
}

struct LSTMParams {
  // The number of features in the lstm hidden state
  1: i32 lstm_dim = 100,
  // The number of lstm layers to use
  2: i32 num_layers = 1,
}

struct CNNParams {
  // Number of feature maps for each kernel
  1: i32 kernel_num = 100,
  // Kernel sizes to use in convolution
  2: list<i32> kernel_sizes = [3, 4],
}

struct DocBLSTM {
  1: float dropout = 0.4,
  // The hidden dimension for the self attention layer
  2: i32 self_attn_dim = 64,
  3: LSTMParams lstm,
}

struct DocNN {
  1: float dropout = 0.4,
  2: CNNParams cnn,
}

struct WordBLSTM {
  1: float dropout = 0.4,
  2: LSTMParams lstm,
  3: bool use_crf = false,
  4: i32 slot_attn_dim = 64,
  // When slot attention is enabled, ONNX export will fail
  5: SlotAttentionType slot_attention_type = NO_ATTENTION,
}

struct WordCNN {
  1: float dropout = 0.4,
  2: CNNParams cnn,
  3: i32 fwd_bwd_context_len = 5,
  4: i32 surrounding_context_len = 2,
  5: bool use_crf = false,
}

enum SlotAttentionType{
  NO_ATTENTION = 0,
  CONCAT = 1,
  MULTIPLY =2 ,
  DOT = 3
}
// Input parameters for the joint document classification and sequence tagging
// model described here: https://fburl.com/2c9s3o62
struct JointBLSTM {
  1: float dropout = 0.4,
  2: i32 self_attn_dim = 64,
  3: LSTMParams lstm,
  // When slot attention is enabled, ONNX export will fail
  4: SlotAttentionType slot_attention_type = NO_ATTENTION,
  5: bool use_doc_probs_in_word = false,
  6: bool use_crf = false,
  7: float default_doc_loss_weight = 0.2,
  8: float default_word_loss_weight = 0.5
}

struct JointCNN {
  1: float dropout = 0.4,
  2: CNNParams cnn,
  3: i32 fwd_bwd_context_len = 5,
  4: i32 surrounding_context_len = 2,
  5: float default_doc_loss_weight = 0.2,
  6: float default_word_loss_weight = 0.5,
  7: bool use_crf = false,
  8: bool use_doc_probs_in_word = true,
}

enum CompositionalType{
  BLSTM = 0,
  SUM = 1,
}

struct AblationParams {
  1: bool use_buffer = true,
  2: bool use_stack = true,
  3: bool use_action = true,
}

struct RNNGConstraints {
  1: bool intent_slot_nesting = true,
  2: bool ignore_loss_for_unsupported = false,
  3: bool no_slots_inside_unsupported = true,
}

// RNNG model specific parameters
struct RNNG {
  1: i32 max_train_num = -1,
  2: i32 max_dev_num = -1,
  3: i32 max_test_num = -1,
  4: i32 max_open_NT = 10,
  5: float dropout = 0.1,
  6: CompositionalType compositional_type = BLSTM,
  7: bool all_metrics = 0,
  8: LSTMParams lstm,
  9: AblationParams ablation,
  10: RNNGConstraints constraints,
  11: bool use_cpp = false,
}

// Temporary, until a native seq2seq implementation in PyText
struct Seq2Seq {
  1: bool transform_to_noLabel = true,
  2: string test_target_path = " ",
  3: string seq2seq_params_path,
}

union Compositional {
  1: RNNG rnng,
  2: Seq2Seq seq2seq,
}

struct PairNN {
  1: PairNNSubModel query_model,
  2: PairNNSubModel result_model,
  3: required i32 embed_size,
  4: optional float margin = 1.0,
}

enum NLGLoss {
  NLLLoss = 0,
  CrossEntropy = 1,
}

struct NNIR {
  1: i32 max_sentence_length = 30,
  2: i32 max_test_num = -1,
  3: i32 max_test_print_num = 10,
  4: i32 max_n_gram = 4,
  5: i32 max_candidates = -1,
  6: bool tokenize_values = true,
}

enum GenerationModelType {
  GENLSTM = 0,
  CONDITIONED_LSTM = 1,
}

struct GenLSTM {
  1: LSTMParams lstm,
  2: i32 max_sentence_length = 30,
  3: i32 max_dev_num = -1,
  4: i32 max_test_num = -1,
  5: i32 max_test_print_num = 10,
  6: NLGLoss loss_type = CrossEntropy,
  7: bool ignore_values = false, // DEPRECATED
  8: bool use_canonical_forms = true,
  9: GenerationModelType model_type = GENLSTM,
  10: bool tokenize_values = true,
  // Parameters specific to training sc-LSTM - see equation 13 in
  // https://arxiv.org/pdf/1508.01745.pdf
  // Constant weight used in loss function for sc-LSTM,
  // used to penalize turning off multiple gates in the same timestep
  11: float eta = 0.0001,
  // Exponentiation base used in loss function for sc-LSTM,
  // used to penalize turning off multiple gates in the same timestep
  12: float zeta = 100,
  // scales component coming from hidden state in sc-LSTM read gate
  13: float alpha = 0.5,
}

struct FeedForwardNN {
  1: i32 input_size = 2400,
  2: i32 hidden_layers = 2,
  3: list<i32> hidden_units = [300, 100],
  4: float dropout = 0.5,
  5: i32 num_classes = 2,
  6: string activation = "relu",
  7: string final_activation = "logsigmoid"
}

struct DepParser {

}

struct LMLSTM {
  1: float dropout = 0.4,
  2: LSTMParams lstm,
  3: bool tied_weights = false,
}

union Model {
  1: DocBLSTM docblstm,
  2: DocNN docnn,
  3: WordBLSTM wordblstm,
  4: JointBLSTM jointblstm,
  5: Compositional compositional,
  6: PairNN pairnn,
  7: GenLSTM genlstm,
  8: NNIR nnir,
  9: WordCNN wordcnn,
  10: DepParser dep_parser,
  // Deprecated 11: JointBLSTMPROD jointblstm_prod,
  // Deprecated 13: WordBLSTMPROD wordblstm_prod,
  14: JointCNN jointcnn,
  15: FeedForwardNN feedforwardnn,
  16: BaggingDocEnsemble bagging_doc_ensemble,
  17: BaggingJointEnsemble bagging_joint_ensemble,
  18: LMLSTM lmlstm,
}

union PairNNSubModel {
  1: DocNN docnn,
}

struct PyTextConfig {
  1: TrainingParams train_params,
  2: DataParams data_params,
  3: Model model,
  4: FeatureParams features_params,
  5: bool use_cuda_if_available = true,
  // If set, then just test the given snapshot and exit
  6: bool test_given_snapshot = 0,
  // A path to a snapshot of a trained model to test
  7: string load_snapshot_path = "",
  8: OutputPaths output_paths,
  // If set, we will try to export the model to a caffe2 model and prepare it
  // for publishing in fblearner predictor, make sure the model is compaitable
  // with ONNX before setting this, otherwise an exception will be thrown
  9: bool export_to_fbl_predictor = false,
  10: DecodingParams decoding_params,
}
