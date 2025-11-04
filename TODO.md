# Project To-Do List

## 🚀 Features
- [x] Setup project structure
- [x] Train Neural Network to detect fibers
- [ ] !Develop criteria for high quality fibers
- [ ] !Implement detection and segmentation of high confidence fiber fragmetns
- [ ] Use python instead of Java for ridge detection
- [ ] Add training support
- [x] Save svs in pixels and units
- [x] Add calculation of Mean and OI
- [x] Diameter shold be calculated based on original not rounded distance transform
- [x] Change printing to logging
- [x] change radiobuttions to drow down list
- [x] add loading bar
- [x] Add gx,gy vizualization
- [x] Build diameter lines based on gradients
- [ ] Add clustering of fiber fragmetns into one fiber
- [ ] Batch processing
- [ ] !Improve dataset and finetune NN
- [ ] Tiling for predictions
- [x] !Add porosity measurement 
- [ ] Add zoom
- [ ] Add scalebar tool (or automatic scalebar detection)
- [ ] Add logbar
- [ ] Use lighweight alternative for torch for inference only
- [ ] Use configuratin file

## 🐞 Bugs
- [x] Fix x,y order for arrow drawing
- [ ] Harmonize everything to x,y order or add _xy postfix
- [ ] Arrows work incorrectly again
- [ ] Save resulting images with original filename
- [ ] Check algorithm of fiber porosity
- [x] Dt pore image is not updated

## 🧹 Refactoring / Cleanup
- [x] Improve architecture so each class owns only specialized functions (eg FiberAnalyzer own all image processing)

## 📝 Documentation
- [ ] Write minimal documentation
- [ ] Create presentation
- [ ] Start writing article


