// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

namespace cpp predictor_service

service Predictor {
   // Returns list of scores for each label
   map<string,list<double>> predict(1:string doc),
}
