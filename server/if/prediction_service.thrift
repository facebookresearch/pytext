include "fblearner/predictor/if/prediction_service.thrift"
include "libfb/py/controller/Controller.thrift"

namespace php pytext_prediction_service
namespace py pytext.server.prediction_service
namespace py3 pytext.server

service PytextPredictionService extends Controller.ControllerService {
  prediction_service.PredictionResponse predict(
    1: prediction_service.PredictionRequest request,
  ) throws (1: prediction_service.PredictionException ex)
}
