import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

/**
 * for SIGMETRICS2020 sketchBlm
 * changes based on CountMin: 
 * 1) one array
 * 2) change in initialization of w (w is the number of basic data structures in each segment, w / m)
 * 3) same encode and estimate
 * @author Youlin
 */

public class GeneralSketchBloom {
	public static Random rand = new Random();

	public static int n = 0; 						// total number of packets
	public static int flows = 0; 					// total number of flows
	public static int avgAccess = 0; 				// average memory access for each packet
	public static  int M = 1024 * 1024* 32; 	// total memory space Mbits	
	public static GeneralDataStructure[][] C;
	public static GeneralDataStructure[][][] CP;
	public static Set<Integer> sizeMeasurementConfig = new HashSet<>(Arrays.asList(0)); // 0-Counter; 1-Bitmap; 2-FM sketch; 3-HLL sketch
	public static Set<Integer> spreadMeasurementConfig = new HashSet<>(Arrays.asList()); // 1-Bitmap; 2-FM sketch; 3-HLL sketch

	/** parameters for count-min */
	public static final int d = 4; 			// the number of rows in Count Min
	public static int w = 1;				// the number of columns in Count Min
	public static int u = 1;				// the size of each elementary data structure in Count Min.
	public static int[] S = new int[d];		// random seeds for Count Min
	public static int m = 1;				// number of bit/register in each unit (used for bitmap, FM sketch and HLL sketch)


	/** parameters for counter */
	public static int mValueCounter = 1;			// only one counter in the counter data structure
	public static int counterSize = 32;				// size of each unit

	/** parameters for bitmap */
	public static  int bitArrayLength = 5000;

	/** parameters for FM sketch **/
	public static int mValueFM = 128;
	public static  int FMsketchSize = 32;

	/** parameters for HLL sketch **/
	public static int mValueHLL = 128;
	public static final int HLLSize = 5;

	public static int times = 1;
	
	/** number of runs for throughput measurement */
	public static int loops = 100;

	public static int[][] Marray= {{8},{8}};
	public static int[][]	mValueCounterarray= {{1},{1}};
	public static int[][] bitArrayLengtharray= {{50000},{5000}};
	public static int[][] mValueFMarray= {{128},{128}};
	public static int[][] mValueHLLarray= {{128},{128}};
	public static int[][] periodsarray= {{1,5,10},{1,5,10}};
	
	public static void main(String[] args) throws FileNotFoundException {
		System.out.println("Start****************************");
		for (int i : sizeMeasurementConfig) {
			for(int l=0;l<periodsarray[0].length;l++) {
				GeneralUtil.periods=periodsarray[0][l];
			for(int i1=0;i1<Marray[0].length;i1++) {
				M=Marray[0][i1]*1024*1024;
				switch (i) {
				case 0:
				   for(int j=0;j<mValueCounterarray[0].length;j++) {
					   mValueCounter=mValueCounterarray[0][j];
					   initCM(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
			        	estimateSize(GeneralUtil.dataSummaryForFlowSize);

				   }
				   break;
				case 1:
					for(int j=0;j<bitArrayLengtharray[0].length;j++) {
						bitArrayLength=bitArrayLengtharray[0][j];
						initCM(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
			        	estimateSize(GeneralUtil.dataSummaryForFlowSize);

					}
					break;
				case 2:	
					for(int j=0;j<mValueFMarray[0].length;j++) {
						mValueFM=mValueFMarray[0][j];
						initCM(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
			        	estimateSize(GeneralUtil.dataSummaryForFlowSize);

					}
					break;
				case 3:
					for(int j=0;j<mValueHLLarray[0].length;j++) {			
						mValueHLL=mValueHLLarray[0][j];
						initCM(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
			        	estimateSize(GeneralUtil.dataSummaryForFlowSize);

					}
					break;
				default:break;
				}
			}
			}
		}
		
		/** measurment for flow spreads **/
		for (int i : spreadMeasurementConfig) {
			for(int l=0;l<periodsarray[1].length;l++) {
				GeneralUtil.periods=periodsarray[1][l];
			for(int i1=0;i1<Marray[1].length;i1++) {
				M=Marray[1][i1]*1024*1024;
				switch (i) {
				case 1:
					for(int j=0;j<bitArrayLengtharray[1].length;j++) {
						bitArrayLength=bitArrayLengtharray[1][j];
						initCM(i);
						encodeSpread(GeneralUtil.dataStreamForFlowSpread);
			    		estimateSpread(GeneralUtil.dataSummaryForFlowSpread);
					}
					break;
				case 2:	
					for(int j=0;j<mValueFMarray[1].length;j++) {
						mValueFM=mValueFMarray[1][j];
						initCM(i);
						encodeSpread(GeneralUtil.dataStreamForFlowSpread);
			    		estimateSpread(GeneralUtil.dataSummaryForFlowSpread);
					}
					break;
				case 3:
					for(int j=0;j<mValueHLLarray[1].length;j++) {			
						mValueHLL=mValueHLLarray[1][j];
						initCM(i);
						encodeSpread(GeneralUtil.dataStreamForFlowSpread);
			    		estimateSpread(GeneralUtil.dataSummaryForFlowSpread);
					}
					break;
				default:break;
				}
			}
			}
		}


		System.out.println("DONE!****************************");
	}

	// Generate counter base Counter Bloom for flow size measurement.
	public static Counter[][] generateCounter() {
		m = mValueCounter;
		u = counterSize * mValueCounter;
		w = (M / u);
		Counter[][] B = new Counter[1][w];
		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < w; j++) {
				B[i][j] = new Counter(1, counterSize);
			}
		}
		
		Counter[][][] BP = new Counter[GeneralUtil.periods][1][w];
		for(int t = 0; t < GeneralUtil.periods; t++) {
			for(int j=0;j<w;j++)
			BP[t][0][j] = new Counter(1, counterSize);
		}
		CP = BP;
		return B;
	}

	// Generate bitmap base Bitmap Bloom for flow cardinality measurement.
	public static Bitmap[][] generateBitmap() {
		m = bitArrayLength;
		u = bitArrayLength;
		w = (M / u) / 1;
		Bitmap[][] B = new Bitmap[1][w];
		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < w; j++) {
				B[i][j] = new Bitmap(bitArrayLength);
			}
		}
		Bitmap[][][] BP = new Bitmap[GeneralUtil.periods][1][w];
		for(int t = 0; t < GeneralUtil.periods; t++) {
			for(int j=0;j<w;j++)
			BP[t][0][j] = new Bitmap(bitArrayLength);
		}
		CP = BP;
		return B;
	}

	// Generate FM sketch base FMsketch Bloom for flow cardinality measurement.
	public static FMsketch[][] generateFMsketch() {
		m = mValueFM;
		u = FMsketchSize * mValueFM;
		w = (M / u) / 1;
		FMsketch[][] B = new FMsketch[1][w];
		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < w; j++) {
				B[i][j] = new FMsketch(mValueFM, FMsketchSize);
			}
		}
		
		FMsketch[][][] BP = new FMsketch[GeneralUtil.periods][1][w];
		for(int t = 0; t < GeneralUtil.periods; t++) {
			for(int j=0;j<w;j++)
			 BP[t][0][j] = new FMsketch(mValueFM, FMsketchSize);
		}
		CP = BP;
		return B;
	}

	// Generate HLL sketch base HLL Bloom for flow cardinality measurement.
	public static HyperLogLog[][] generateHyperLogLog() {
		m = mValueHLL;
		u = HLLSize * mValueHLL;
		w = (M / u) / 1;
		HyperLogLog[][] B = new HyperLogLog[1][w];
		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < w; j++) {
				B[i][j] = new HyperLogLog(mValueHLL, HLLSize);
			}
		}
		HyperLogLog[][][] BP = new HyperLogLog[GeneralUtil.periods][1][w];
		for(int t = 0; t < GeneralUtil.periods; t++) {
			for(int j=0;j<w;j++)
			BP[t][0][j] = new HyperLogLog(mValueHLL, HLLSize);
		}
		CP = BP;
		return B;
	}

	// Init the Count Min for different elementary data structures.
	public static void initCM(int index) {
		switch (index) {
		case 0: case -1: C = generateCounter();
		break;
		case 1:  C = generateBitmap();
		break;
		case 2:  C = generateFMsketch();
		break;
		case 3:  C = generateHyperLogLog();
		break;
		default: break;
		}
		generateCMRandomSeeds();
		System.out.println("\nSketchBloom-" + C[0][0].getDataStructureName() + " Initialized!-----------");
	}
	
	// Generate random seeds for Counter Min.
	public static void generateCMRandomSeeds() {
		HashSet<Integer> seeds = new HashSet<Integer>();
		int num = d;
		while (num > 0) {
			int s = rand.nextInt();
			if (!seeds.contains(s)) {
				num--;
				S[num] = s;
				seeds.add(s);
			}
		}
	}

	/** Encode elements to the Count Min for flow size measurement. */
	public static void encodeSize(String filePath) throws FileNotFoundException {
		System.out.println("Encoding elements using " + C[0][0].getDataStructureName().toUpperCase() + "s for flow size measurement......" );
		//Scanner sc = new Scanner(new File(filePath));
		n = 0;
		for (int t = 0; t < GeneralUtil.periods; t++) {
			Scanner sc = new Scanner(new File(filePath + "output"+t+"v.txt"));

			System.out.println("Input file: " + filePath + "output"+t+"v.txt" );

		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			String flowid = GeneralUtil.getSizeFlowID(strs, true);
			n++;

			/*/if (C[0][0].getDataStructureName().equals("Counter")) {
				int minVal = Integer.MAX_VALUE;
				for (int i = 0; i < d; i++) {
					int j = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowid) ^ S[i]) % w + w) % w;
					minVal = Math.min(minVal, C[0][j].getValue());
				}
				for (int i = 0; i < d; i++) {
					int j = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowid) ^ S[i]) % w + w) % w;
					if (C[0][j].getValue() == minVal) {
						C[0][j].encode();           
					}
				}
			} else {/*/
	
				for (int i = 0; i < d; i++) {
					int j = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowid) ^ S[i]) % w + w) % w; 
					CP[t][0][j].encode();
			//	}
			}
		}
		System.out.println("Total number of encoded pakcets: " + n);
		sc.close();
		}
		for (int t = 0; t < GeneralUtil.periods; t++) {
			for(int j=0;j<w;j++)
			C[0][j] = C[0][j].join(CP[t][0][j]);
		}
	}
    
	
	/** Estimate flow sizes. */
	public static void estimateSize(String filePath) throws FileNotFoundException {
		System.out.println("Estimating Flow SIZEs..." ); 
		Scanner sc = new Scanner(new File(filePath + GeneralUtil.periods + "\\outputsrcDstCount.txt"));
		System.out.println(filePath + GeneralUtil.periods + "\\outputsrcDstCount.txt");
		String resultFilePath = GeneralUtil.path + "SketchBloom\\size\\" + C[0][0].getDataStructureName()
				+ "_M_" +  M / 1024 / 1024 + "_d_" + d + "_u_" + u + "_m_" + m + "_TTT_" + GeneralUtil.periods;
		PrintWriter pw = new PrintWriter(new File(resultFilePath));
		System.out.println("w :" + w);
		System.out.println("Result directory: " + resultFilePath); 
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			String flowid = GeneralUtil.getSizeFlowID(strs, false);
			int num = Integer.parseInt(strs[strs.length-1]);

				int estimate = Integer.MAX_VALUE;

				for(int i = 0; i < d; i++) {
					int j = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowid) ^ S[i]) % w + w) % w;
					estimate = Math.min(estimate, C[0][j].getValue());
				}
				pw.println(entry + "\t" + estimate);
		}
		sc.close();
		pw.close();
	}
	
	/** Encode elements to the Count Min for flow spread measurement. */
	public static void encodeSpread(String filePath) throws FileNotFoundException {
		System.out.println("Encoding elements using " + C[0][0].getDataStructureName().toUpperCase() + "s for flow spread measurement......" );
		//Scanner sc = new Scanner(new File(filePath));
		n = 0;
		for (int t = 0; t < GeneralUtil.periods; t++) {
			Scanner sc = new Scanner(new File(filePath + "output"+t+"v.txt"));
			
			System.out.println("Input file: " + filePath + "output"+t+"v.txt" );
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			String[] res = GeneralUtil.getSperadFlowIDAndElementID(strs, true);
			String flowid = res[0];
			String elementid = res[1];
			n++;
			for (int i = 0; i < d; i++) {
				int j = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowid) ^ S[i]) % w + w) % w;
				CP[t][0][j].encode(elementid);
			}
		}
		System.out.println("Total number of encoded pakcets: " + n); 
		sc.close();
		}
		for (int t = 0; t < GeneralUtil.periods; t++) {
			for(int j=0;j<w;j++)
			C[0][j] = C[0][j].join(CP[t][0][j]);
		}
	}

	/** Estimate flow spreads. */
	public static void estimateSpread(String filepath) throws FileNotFoundException {
		System.out.println("Estimating Flow CARDINALITY..." ); 
		Scanner sc = new Scanner(new File(filepath + GeneralUtil.periods + "\\outputdstCard.txt"));
		System.out.println(filepath + GeneralUtil.periods + "\\outputdstCard.txt");
		String resultFilePath = GeneralUtil.path + "SketchBloom\\spread\\" + C[0][0].getDataStructureName()
				+ "_M_" +  M / 1024 / 1024 + "_d_" + d + "_u_" + u + "_m_" + m+"_TTT_" + GeneralUtil.periods;
		PrintWriter pw = new PrintWriter(new File(resultFilePath));
		System.out.println("Result directory: " + resultFilePath); 
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			String flowid = GeneralUtil.getSperadFlowIDAndElementID(strs, false)[0];
			int num = Integer.parseInt(strs[strs.length-1]);
			
				int estimate = Integer.MAX_VALUE;
				
				for(int i = 0; i < d; i++) {
					int j = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowid) ^ S[i]) % w + w) % w;
					estimate = Math.min(estimate, C[0][j].getValue());
				}
				pw.println(entry + "\t" + estimate);
		}
		sc.close();
		pw.close();

	}
	
}
