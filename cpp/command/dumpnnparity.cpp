#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/rand.h"
#include "../dataio/numpywrite.h"
#include "../game/randomopening.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

//Evaluates the model on random positions and writes the inputs and the postprocessed outputs to an npz file.
//python/test_nn_parity.py compares them with the python model, or two such files (different backends) with each other.
int MainCmds::dumpnnparity(const vector<string>& args) {
  Board::initHash();

  ConfigParser cfg;
  string modelFile;
  string outFile;
  int numPositions;
  int nnLen;
  string seed;
  try {
    KataGoCommandLine cmd("Dump neural net inputs and outputs on random positions, to check backends against each other and against python.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> outArg("","out","Output npz file",true,string(),"FILE");
    TCLAP::ValueArg<int> numPositionsArg("","num-positions","Number of positions, default 200",false,200,"N");
    TCLAP::ValueArg<int> nnLenArg("","nn-len","Neural net board length, default 11",false,11,"LEN");
    TCLAP::ValueArg<string> seedArg("","seed","Random seed for the positions",false,"dumpnnparity","SEED");
    cmd.add(outArg);
    cmd.add(numPositionsArg);
    cmd.add(nnLenArg);
    cmd.add(seedArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    outFile = outArg.getValue();
    numPositions = numPositionsArg.getValue();
    nnLen = nnLenArg.getValue();
    seed = seedArg.getValue();
    cmd.getConfig(cfg);
    if(numPositions < 1)
      throw StringError("num-positions must be positive");
    if(nnLen < 5 || nnLen > NNPos::MAX_BOARD_LEN)
      throw StringError("Invalid nn-len");
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(NULL, true, false, false);
  Rand seedRand;
  const int maxConcurrentEvals = 16;
  const int expectedConcurrentEvals = 1;
  const int defaultMaxBatchSize = 1;
  const bool defaultRequireExactNNLen = false;
  const bool disableFP16 = false;
  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
    modelFile,modelFile,"",cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
    nnLen,nnLen,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
    Setup::SETUP_FOR_BENCHMARK
  );

  const int modelVersion = nnEval->getModelVersion();
  const int inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  const int numSpatial = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobal = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int policySize = NNPos::getPolicySize(nnLen,nnLen);
  if(inputsVersion != 102 && inputsVersion != 103)
    throw StringError("Unsupported inputs version " + Global::intToString(inputsVersion));

  NumpyBuffer<float> spatialBuf({numPositions, numSpatial, nnLen, nnLen});
  NumpyBuffer<float> globalBuf({numPositions, numGlobal});
  //Policy probs, -1 for illegal moves, pass last
  NumpyBuffer<float> policyBuf({numPositions, policySize});
  //Win, loss, no result for the player to move
  NumpyBuffer<float> valueBuf({numPositions, 3});
  //movesLeft, microMovesLeft, varTimeLeft, shorttermWinlossError
  NumpyBuffer<float> miscBuf({numPositions, 4});

  Rand rand(seed);
  int count = 0;
  int64_t numEvals = 0;
  while(count < numPositions) {
    int size = rand.nextBool(0.25) ? 5 + (int)rand.nextUInt(nnLen - 4) : nnLen;
    Board board = RandomOpening::initStartPosition(size, size, rand, 0.5, 0.5, 0.3);
    Rules rules(rand.nextBool(0.5) ? 0 : 20 + (int)rand.nextUInt(200));
    rules.forbidPassUnlessNoOtherMove = rand.nextBool(0.3);
    BoardHistory hist(board, board.nextPla, rules);

    //Sample a few positions from each random game
    int numMoves = (int)rand.nextUInt(2000);
    for(int m = 0; m <= numMoves && count < numPositions; m++) {
      if(hist.isGameFinished)
        break;
      Player pla = board.nextPla;
      if(m == numMoves || rand.nextBool(0.01)) {
        MiscNNInputParams params;
        params.symmetry = 0;
        NNResultBuf buf;
        const bool skipCache = true;
        nnEval->evaluate(board, hist, pla, params, buf, skipCache);
        numEvals++;

        float* rowSpatial = spatialBuf.data + (size_t)count * numSpatial * nnLen * nnLen;
        float* rowGlobal = globalBuf.data + (size_t)count * numGlobal;
        if(inputsVersion == 102)
          NNInputs::fillRowV102(board, hist, pla, params, nnLen, nnLen, false, rowSpatial, rowGlobal);
        else
          NNInputs::fillRowV103(board, hist, pla, params, nnLen, nnLen, false, rowSpatial, rowGlobal);

        for(int i = 0; i < policySize; i++)
          policyBuf.data[(size_t)count * policySize + i] = buf.policyResult[i];
        const NNOutput& out = *(buf.result);
        float* rowValue = valueBuf.data + (size_t)count * 3;
        rowValue[0] = pla == P_WHITE ? out.whiteWinProb : out.whiteLossProb;
        rowValue[1] = pla == P_WHITE ? out.whiteLossProb : out.whiteWinProb;
        rowValue[2] = out.whiteNoResultProb;
        float* rowMisc = miscBuf.data + (size_t)count * 4;
        rowMisc[0] = out.movesLeft;
        rowMisc[1] = out.microMovesLeft;
        rowMisc[2] = out.varTimeLeft;
        rowMisc[3] = out.shorttermWinlossError;
        count++;
      }

      //Random legal move, pass less often so that games get somewhere
      vector<Loc> moves;
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x, y, board.x_size);
          if(hist.isLegal(board, loc, pla))
            moves.push_back(loc);
        }
      }
      bool passLegal = hist.isLegal(board, Board::PASS_LOC, pla);
      Loc loc;
      if(moves.size() == 0 || (passLegal && rand.nextBool(0.1)))
        loc = Board::PASS_LOC;
      else
        loc = moves[rand.nextUInt((uint32_t)moves.size())];
      hist.makeBoardMoveAssumeLegal(board, loc, pla);
    }
  }

  ZipFile zipFile(outFile);
  uint64_t numBytes;
  numBytes = spatialBuf.prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("binaryInputNCHW", spatialBuf.dataIncludingHeader, numBytes);
  numBytes = globalBuf.prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("globalInputNC", globalBuf.dataIncludingHeader, numBytes);
  numBytes = policyBuf.prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("policyProbs", policyBuf.dataIncludingHeader, numBytes);
  numBytes = valueBuf.prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("value", valueBuf.dataIncludingHeader, numBytes);
  numBytes = miscBuf.prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("misc", miscBuf.dataIncludingHeader, numBytes);
  zipFile.close();

  logger.write(
    "Wrote " + Global::intToString(count) + " positions (model version " + Global::intToString(modelVersion) +
    ", inputs version " + Global::intToString(inputsVersion) + ") to " + outFile);

  delete nnEval;
  NeuralNet::globalCleanup();
  return 0;
}
