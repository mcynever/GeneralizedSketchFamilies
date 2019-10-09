#include "GeneralSketchBloom.h"

/** parameters for count-min */
const int d = 4; 			// the nubmer of rows in Count Min

GeneralSketchBloom *initSketchBloom(int sketchName)
{
	int M = 1024 * 1024 * 4; 	// total memory space Mbits	

	
	/** parameters for counter */
	const int mValueCounter = 1;			// only one counter in the counter data structure
	const int counterSize = 32;				// size of each unit

	/** parameters for bitmap */
	const int bitArrayLength = 20000;

	/** parameters for FM sketch **/
	const int mValueFM = 128;
	const int FMsketchSize = 32;

	/** parameters for HLL sketch **/
	const int mValueHLL = 128;
	const int HLLSize = 5;

	GeneralSketchBloom *GSB = new(GeneralSketchBloom);
	GSB->sketchName = sketchName;
	if (sketchName == 0) {
		GSB->m = mValueCounter;
		GSB->u = counterSize;
		int w = M / mValueCounter / counterSize;
		GSB->w = w;
		GSB->C = (Counter ***)malloc(sizeof(Counter**));
		int i = 0;
		for (i = 0; i < 1; i++) {
			GSB->C[0] = (Counter **)malloc(GSB->w * sizeof(Counter*));
			int j;
			for (j = 0; j < GSB->w; j++) {
				GSB->C[0][j] = newCounter(GSB->m, GSB->u);
			}
		}
		GSB->instance = GSB->C;
		printf("%s\n", "\nGeneral SketchBloom-Counter Initialized!");
	} else if (sketchName == 1) {
		GSB->m = bitArrayLength;
		GSB->u = bitArrayLength;
		GSB->w = (M / GSB->u) / 1;
		GSB->B = (Bitmap ***)malloc(sizeof(Bitmap **));
		int i = 0;
		for (i = 0; i < 1; i++) {
			GSB->B[0] = (Bitmap **)malloc(GSB->w * sizeof(Bitmap *));
			int j;
			for (j = 0; j < GSB->w; j++)
				GSB->B[0][j] = newBitmap(GSB->m, GSB->u);
		}
		printf("%s\n", "\nGeneral SketchBloom-Bitmap Initialized!");
	} else if (sketchName == 2) {
		GSB->m = mValueFM;
		GSB->u = FMsketchSize;
		GSB->w = (M / GSB->u / GSB->m) / 1;
		GSB->F = (FMsketch ***)malloc(sizeof(FMsketch**));
		int i;
		for (i = 0; i < 1; i++) {
			GSB->F[0] = (FMsketch **)malloc(GSB->w * sizeof(FMsketch*));
			int j;
			for (j = 0; j < GSB->w; j++) {
				GSB->F[0][j] = newFMsketch(GSB->m, GSB->u);
			}
		}
		printf("%s\n", "\nGeneral SketchBloom-FMsketch Initialized!");
	} else if (sketchName == 3) {
		GSB->m = mValueHLL;
		GSB->u = HLLSize;
		GSB->w = (M / (GSB->u * GSB->m)) / 1;
		GSB->H = (HyperLogLog ***)malloc(sizeof(HyperLogLog**));
		int i = 0;
		for (i = 0; i < 1; i++) {
			GSB->H[0] = (HyperLogLog **)malloc(GSB->w * sizeof(HyperLogLog*));
			printf("1!\n");
			int j;
			for (j = 0; j < GSB->w; j++) {
				//printf("2!\n");
				GSB->H[0][j] = newHyperLogLog(GSB->m, GSB->u);
			}
		}
		printf("%s\n", "\nGeneral SketchBloom-HyperLogLog Initialized!");
	}
	else {
		printf("%s\n", "Unsupported Data Structure!!");
	}
	generateSketchBloomRandomSeeds(GSB);
	return GSB;
}

void generateSketchBloomRandomSeeds(GeneralSketchBloom *GSB)
{
	int num = d;
	GSB->S = newArr(int, d);
	int i = 0;
	for (i = 0; i < d; i++) {
		*(GSB->S + i) = rand();
	}
}

//void getThroughput()
//{
//	printf("%s\n", "Measuring Throughtput...");
//	int *dataFlowID, *dataElemID;
//	string resultFilePath = GeneralUtil::dataStreamForFlowThroughput;
//	char c[10 * 1024];
//	const char *filename = resultFilePath.c_str();
//	FILE *fp;
//	fopen_s(&fp, filename, "r");
//	if (!fp) {
//		cout << "file not existed!" << endl;
//		exit(1);
//	}
//
//	char delimiter = '\t';
//
//	if (sizeMeasurementConfig.size() > 0) {
//		int lineNo = 0;
//		while (fgets(c, sizeof(c), fp) != NULL) {
//			c[strlen(c) - 1] = '\0';
//			string s(c);
//			vector<string> strs;
//			int i = 0;
//			while (i < s.size()) {
//				while (s[i] == delimiter && i < s.size()) {
//					i++;
//				}
//				int start = i;
//				while (s[i] != delimiter && i < s.size()) {
//					i++;
//				}
//				if (i != start) {
//					strs.push_back(s.substr(start, i - start));
//				}
//			}
//			//TODO: can be removed
//			if (lineNo % 100000 == 0) {
//				cout << lineNo << endl;
//			}
//
//			lineNo++;
//			dataFlowID.push_back(GeneralUtil::FNVHash1(strs[0] + "\t" + strs[1]));
//			dataElemID.push_back(GeneralUtil::FNVHash1(strs[1]));
//		}
//	}
//	else {
//		int lineNo = 0;
//		while (fgets(c, sizeof(c), fp) != NULL) {
//			c[strlen(c) - 1] = '\0';
//			string s(c);
//			vector<string> strs;
//			int i = 0;
//			while (i < s.size()) {
//				while (s[i] == delimiter && i < s.size()) {
//					i++;
//				}
//				int start = i;
//				while (s[i] != delimiter && i < s.size()) {
//					i++;
//				}
//				if (i != start) {
//					strs.push_back(s.substr(start, i - start));
//				}
//			}
//			//TODO: can be removed
//			if (lineNo % 100000 == 0) {
//				cout << lineNo << endl;
//			}
//
//			lineNo++;
//			dataFlowID.push_back(GeneralUtil::FNVHash1(strs[0]));
//			dataElemID.push_back(GeneralUtil::FNVHash1(strs[1]));
//		}
//	}
//
//	/** measurment for flow sizes **/
//	for (int i : sizeMeasurementConfig) {
//		throughputForSize(i, dataFlowID, dataElemID);
//	}
//
//	/** measurment for flow spreads **/
//	for (int i : spreadMeasurementConfig) {
//		throughputForSpread(i, dataFlowID, dataElemID);
//	}
//}

//void throughputForSize(int sketchName, int *dataFlowID, int *dataElemID)
//{
//	printf("%s\n", "Calculating throughput for size measurement...");
//	int totalNum = dataFlowID.size();
//	initSketchBloom(sketchName);
//	string resultFilePath = GeneralUtil::path + "\\Throughput\\SketchBloom_size_" + GeneralUtil::getDataStructureName(sketchName)
//		+ "_M_" + to_string(M / 1024 / 1024) + "_d_" + to_string(d) + "_u_" + to_string(u) + "_m_" + to_string(m) + "_tp_" + to_string(GeneralUtil::throughputSamplingRate);
//	ofstream fout;
//	fout.open(resultFilePath);
//	if (fout.fail()) {
//		cout << "output file not existed" << endl;
//		exit(1);
//	}
//	double res = 0.0;
//
//	if (sketchName == 0) { // for enhanced countmin
//		auto duration = 0.0;
//
//		for (int i = 0; i < loops; i++) {
//			initSketchBloom(sketchName);
//			vector<int> arrIndex(d, 0), arrVal(d, 0);
//
//			high_resolution_clock::time_point t1 = high_resolution_clock::now();
//			for (int j = 0; j < totalNum; j++) {
//				//if (rand.nextDouble() <= GeneralUtil.throughputSamplingRate) {
//				int minVal = INT_MAX;
//				for (int k = 0; k < d; k++) {
//					int jj = (GeneralUtil::intHash(dataFlowID[j] ^ S[k]) % w + w) % w;
//					arrIndex[k] = jj;
//					arrVal[k] = C[0][jj]->getValue();
//					minVal = min(minVal, arrVal[k]);
//				}
//
//				for (int k = 0; k < d; k++) {
//					if (arrVal[k] == minVal) {
//						C[0][arrIndex[k]]->encode(dataElemID[j]);
//					}
//				}
//				//}
//			}
//			high_resolution_clock::time_point t2 = high_resolution_clock::now();
//			duration = duration + duration_cast<nanoseconds>(t2 - t1).count();
//		}
//		duration = duration / 1000000000;
//		res = 1.0 * totalNum / (duration / loops);
//		//System.out.println("Average execution time: " + 1.0 * duration / loops + " seconds");
//		cout << C[0][0]->getDataStructureName() << " Average Throughput: " << 1.0 * totalNum / (duration / loops) << " packets/second" << endl;
//	}
//	else {
//		auto duration = 0.0;
//
//		for (int i = 0; i < loops; i++) {
//			initSketchBloom(sketchName);
//			high_resolution_clock::time_point t1 = high_resolution_clock::now();
//			for (int j = 0; j < totalNum; j++) {
//				//if (rand.nextDouble() <= GeneralUtil.throughputSamplingRate) {
//				for (int k = 0; k < d; k++) {
//					C[0][(GeneralUtil::intHash(dataFlowID[j] ^ S[k]) % w + w) % w]->encode();
//				}
//			}
//			high_resolution_clock::time_point t2 = high_resolution_clock::now();
//			duration = duration + duration_cast<nanoseconds>(t2 - t1).count() * 1.0 / 1000000000;
//		}
//		res = 1.0 * totalNum / (duration / loops);
//		//System.out.println("Average execution time: " + 1.0 * duration / loops + " seconds");
//		cout << C[0][0]->getDataStructureName() << " Average Throughput: " << 1.0 * totalNum / (duration / loops) << " packets/second" << endl;
//	}
//	fout << res << endl;
//	fout.close();
//}
//
//void throughputForSpread(int sketchName, int *dataFlowID, int *dataElemID)
//{
//	cout << "Calculating throughput for spread measurement..." << endl;
//	int totalNum = dataFlowID.size();
//	initSketchBloom(sketchName);
//	string resultFilePath = GeneralUtil::path + "\\Throughput\\SketchBloom_spread_" + GeneralUtil::getDataStructureName(sketchName)
//		+ "_M_" + to_string(M / 1024 / 1024) + "_d_" + to_string(d) + "_u_" + to_string(u) + "_m_" + to_string(m) + "_tp_" + to_string(GeneralUtil::throughputSamplingRate);
//	ofstream fout;
//	fout.open(resultFilePath);
//	if (fout.fail()) {
//		cout << "output file not existed" << endl;
//		exit(1);
//	}
//	double res = 0.0;
//
//	auto duration = 0.0;
//
//	for (int i = 0; i < loops; i++) {
//		initSketchBloom(sketchName);
//		high_resolution_clock::time_point t1 = high_resolution_clock::now();
//		for (int j = 0; j < totalNum; j++) {
//			for (int k = 0; k < d; k++) {
//				C[0][(GeneralUtil::intHash(dataFlowID[i] ^ S[k]) % w + w) % w]->encode(dataElemID[j]);
//			}
//		}
//		high_resolution_clock::time_point t2 = high_resolution_clock::now();
//		duration = duration + duration_cast<nanoseconds>(t2 - t1).count();
//	}
//	duration = duration / 1000000000;
//	res = 1.0 * totalNum / (duration / loops);
//	//System.out.println("Average execution time: " + 1.0 * duration / loops + " seconds");
//	cout << C[0][0]->getDataStructureName() << " Average Throughput: " << 1.0 * totalNum / (duration / loops) << " packets/second" << endl;
//	fout << res << endl;
//	fout.close();
//}
