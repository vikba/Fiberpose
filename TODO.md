# Project To-Do List

## 🚀 Features
- [x] Setup project structure
- [x] Train Neural Network to detect fibers
- [x] !Develop criteria for high quality fibers
- [x] !Implement detection and segmentation of high confidence fiber fragmetns
- [x] Use python instead of Java for ridge detection
- [x] Save svs in pixels and units
- [x] Add calculation of Mean and OI
- [x] Diameter shold be calculated based on original not rounded distance transform
- [x] Change printing to logging
- [x] change radiobuttions to drow down list
- [x] add loading bar
- [x] Add gx,gy vizualization
- [x] Build diameter lines based on gradients
- [x] !Add clustering of fiber fragmetns into one fiber
- [x] Batch processing
- [ ] !Improve dataset and finetune NN
- [x] !Add porosity measurement 
- [x] Add zoom
- [x] Add scalebar tool (or automatic scalebar detection)
- [x] Add default configuration variable
- [x] Load configuration from file
- [x] Create class FiberSegment
- [x] Add plotprofile
- [x] Dynamically add checkpoints from checkpoints folder
- [x] Add simple centerline (ridge) segmentation
- [ ] Implement possibility of pyramid analysis for thin fibers 
- [ ] Tiling for predictions
- [x] Add lightweight ONNX inference runtime
- [x] Allow saving style vector
- [x] Use color coding for angle visialization 
- [x] Add 3D view
- [x] Algorithms for 3D analysis
- [ ] Add checkbox to turn on/off auto brightness/darkness
- [ ] Implement 3-stage cache approach
- [ ] Progress bar
- [ ] For folder analysis add summary file

## 🐞 Bugs
- [x] Fix x,y order for arrow drawing
- [ ] Harmonize everything to x,y order or add _xy postfix
- [x] Arrows work incorrectly again
- [x] Save resulting images with original filename
- [ ] Check algorithm of fiber porosity
- [x] Dt pore image is not updated
- [x] Don't predict dt, if it was already predicted for that image during change of segmetnation method
- [x] Angle distribution is reversed in comparison with Directionality and OrientationJ
- [x] Plot profile wrong when image is scaled

## 🧹 Refactoring / Cleanup
- [x] Improve architecture so each class owns only specialized functions (eg FiberAnalyzer own all image processing)
- [x] Remove excessive hasattr guards
- [x] Re-write compute heavy tasks in numba
- [ ] Refactor process_centerlines
- [x] Refactor compute_centerlines
- [x] Refactor gui_mainwindow
- [ ] Controller QThread
- [ ] Speed up the code


## 📝 Documentation
- [ ] Write minimal documentation
- [ ] Create presentation
- [ ] Start writing article


