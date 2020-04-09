// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

namespace cpp predictor_service

service Predictor {
   // Returns scores for each class
   map<string,double> predict(1:string doc),
}
