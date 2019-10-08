// vSkt improve based on vSketch
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

/**
 * for SIGMETRICS2020 VSKETCH (Segment sharing)
 * changes based on sharing:
 * 1) virtual segment index
 * 2) change in initialization of w (w is the number of basic data structures in each segment, w / m)
 * 3) different encode and estimate due to segment
 * @author Youlin
 */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

public class GeneralvSkt {
public static Random rand = new Random();
	
	public static int n = 0; 						// total number of packets
	public static int flows = 0; 					// total number of flows
	public static int avgAccess = 0; 				// average memory access for each packet
	public static int M = 1024 * 1024 * 16; 	// total memory space Mbits
	public static GeneralDataStructure[] C;
	public static GeneralDataStructure[] B;
	public static Set<Integer> sizeMeasurementConfig = new HashSet<>(Arrays.asList(0)); // 0-counter; 1-Bitmap; 2-FM sketch; 3-HLL sketch
	public static Set<Integer> spreadMeasurementConfig = new HashSet<>(Arrays.asList(3)); // 1-Bitmap; 2-FM sketch; 3-HLL sketch
	
	/** parameters for sharing approach **/
	public static int u = 1;			// the unit size of each virtual data structure
	public static int w;				// number of the unit data structures in the physical array
	public static int m = 1;			// number of elementary data structures in the virtual data structure
	public static int[] S; 				// random seeds for the sharing approach
	
	/** parameters for counters **/
	public static int mValueCounter = 128;
	public static int counterSize = 20;

	/** parameters for bitmap **/
	public static int bitmapSize = 1;	// sharing at bit level
	public static int virtualArrayLength = 10000;
	
	/** parameters for FM sketch **/
	public static int mValueFM = 128;
	public static int FMsketchSize = 32;
	
	/** parameters for hyperLogLog **/
	public static int mValueHLL = 128;
	public static int HLLSize = 5;
	
	public static int times = 1;
	
	/** number of runs for throughput measurement */
	public static int loops = 10;
	public static int Mbase=1024*1024;
	public static int[][] Marray= {{2,4,8,16},{300}};
	public static int[][]	mValueCounterarray= {{1},{1}};
	public static int[][] virtualArrayLengtharray= {{50000},{5000}};
	public static int[][] mValueFMarray= {{64},{128}};
	public static int[][] mValueHLLarray= {{128},{512}};
	public static long c1=0;
	public static long c2=0;
	public static void main(String[] args) throws FileNotFoundException {
		/** measurment for flow sizes **/
		System.out.println("Start************************");
		Random random=new Random();
		c1=random.nextLong();
		c2=random.nextLong();
		/** measurment for flow sizes **/
		for (int i : sizeMeasurementConfig) {
			for(int i1=0;i1<Marray[0].length;i1++) {
				M=Marray[0][i1]*Mbase;
				switch (i) {
				case 0:
				   for(int j=0;j<mValueCounterarray[0].length;j++) {
					   mValueCounter=mValueCounterarray[0][j];
					   initSharing(i);
					   initJoining(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
			        	estimateSize(GeneralUtil.dataSummaryForFlowSize);
				   }
				   break;
				case 1:
					for(int j=0;j<virtualArrayLengtharray[0].length;j++) {
						virtualArrayLength=virtualArrayLengtharray[0][j];
						initSharing(i);
						initJoining(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
			        	estimateSize(GeneralUtil.dataSummaryForFlowSize);
					}
					break;
				case 2:	
					for(int j=0;j<mValueFMarray[0].length;j++) {
						mValueFM=mValueFMarray[0][j];
						initSharing(i);
						initJoining(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
			        	estimateSize(GeneralUtil.dataSummaryForFlowSize);
					}
					break;
				case 3:
					for(int j=0;j<mValueHLLarray[0].length;j++) {			
						mValueHLL=mValueHLLarray[0][j];
						initSharing(i);
						initJoining(i);
						encodeSize(GeneralUtil.dataStreamForFlowSize);
						estimateSize(GeneralUtil.dataSummaryForFlowSize);
					}
					break;
				default:break;
				}
			}
		}
		
		/** measurment for flow spreads **/
		for (int i : spreadMeasurementConfig) {
			for(int i1=0;i1<Marray[1].length;i1++) {
				M=Marray[1][i1]*Mbase;
				switch (i) {
				case 1:
					for(int j=0;j<virtualArrayLengtharray[1].length;j++) {
						virtualArrayLength=virtualArrayLengtharray[1][j];
						initSharing(i);
						initJoining(i);
						encodeSpread(GeneralUtil.dataStreamForFlowSpread);
			    		estimateSpread(GeneralUtil.dataSummaryForFlowSpread);
					}
					break;
				case 2:	
					for(int j=0;j<mValueFMarray[1].length;j++) {
						mValueFM=mValueFMarray[1][j];
						initSharing(i);
						initJoining(i);
						encodeSpread(GeneralUtil.dataStreamForFlowSpread);
			    		estimateSpread(GeneralUtil.dataSummaryForFlowSpread);
					}
					break;
				case 3:
					for(int j=0;j<mValueHLLarray[1].length;j++) {			
						mValueHLL=mValueHLLarray[1][j];
						initSharing(i);
						initJoining(i);
						encodeSpread(GeneralUtil.dataStreamForFlowSpread);
			    		estimateSpread(GeneralUtil.dataSummaryForFlowSpread);
					}
					break;
				default:break;
				}
			}
			
		}
		System.out.println("DONE!!!!!!!!!!!");
	}
	
	// Init the VSketch approach for different elementary data structures.
	public static void initSharing(int index) {
		switch (index) {
	        case 0:  C = generateCounter(); 
	                 break;
	        case 1:  C = generateBitmap(); 
	                 break;
	        case 2:  C = generateFMsketch(); 
	                 break;
	        case 3:  C = generateHyperLogLog();
	                 break;
	        default: break;
		}
		generateSharingRandomSeeds();
		System.out.println("\nVSkt-" + C[0].getDataStructureName() + " Initialized!");
	}
	
	public static void initJoining(int index) {
		switch (index) {
	        case 0:  B = new Counter[1]; B[0]=new Counter(mValueCounter,counterSize); 
	                 break;
	        case 1:  B = new Bitmap[1]; B[0] = new Bitmap(virtualArrayLength);
	                 break;
	        case 2:  B = new FMsketch[1]; B[0] = new FMsketch(mValueFM, FMsketchSize);
	                 break;
	        case 3:  B = new HyperLogLog[1]; B[0] = new HyperLogLog(mValueHLL, HLLSize);
	                 break;
	        default: break;
		}
		//generateSharingRandomSeeds();
		System.out.println("\nVSkt-" + C[0].getDataStructureName() + " Initialized!");
	}
	
	// Generate counter sharing data structures for flow size measurement.
	public static Counter[] generateCounter() {
		m = mValueCounter;
		u = counterSize;
		w = M / u;
		Counter[] B = new Counter[1];
		B[0] = new Counter(w, counterSize);
		return B;
	}
		
	// Generate bit sharing data structures for flow cardinality measurement.
	public static Bitmap[] generateBitmap() {
		m = virtualArrayLength;
		u = bitmapSize;
		w = M / u;
		Bitmap[] B = new Bitmap[1];
		B[0] = new Bitmap(w);
		return B;
	}
	
	// Generate FM sketch sharing data structures for flow cardinality measurement.
	public static FMsketch[] generateFMsketch() {
		m = mValueFM;
		u = FMsketchSize;
		w = M /u; 
		FMsketch[] B = new FMsketch[1];
		B[0] = new FMsketch(w, FMsketchSize);
		return B;
	}
	
	// Generate register sharing data structures for flow cardinality measurement.
	public static HyperLogLog[] generateHyperLogLog() {
		m = mValueHLL;
		u = HLLSize;
		w = M / u;
		HyperLogLog[] B = new HyperLogLog[1];
		B[0] = new HyperLogLog(w, HLLSize);
		return B;
	}
	
	public static HyperLogLog[] generateHyperLogLog(int m,int HLLSize) {
		HyperLogLog[] B = new HyperLogLog[1];
		B[0] = new HyperLogLog(m, HLLSize);
		return B;
	}
	
	// Generate random seeds for Sharing approach.
	public static void generateSharingRandomSeeds() {
		HashSet<Integer> seeds = new HashSet<Integer>();
		S = new int[m];
		int num = m;
		while (num > 0) {
			int s = rand.nextInt();
			if (!seeds.contains(s)) {
				num--;
				S[num] = s;
				seeds.add(s);
			}
		}
	}

	/** Encode elements to the physical data structure for flow size measurement. */
	public static void encodeSize(String filePath) throws FileNotFoundException {
		System.out.println("Encoding elements using " + C[0].getDataStructureName().toUpperCase() + "s for flow size measurement..." );
		Scanner sc = new Scanner(new File(filePath));
		n = 0;
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			long flowid = GeneralUtil.getSize1FlowID(strs, true);
	        n++;
			// segment encode, number of segments
	        C[0].encodeSegment(flowid, S, w / m);
		}
		System.out.println("Total number of encoded pakcets: " + n);
		sc.close();
	}

	/** Estimate flow sizes. */
	public static void estimateSize(String filePath) throws FileNotFoundException {
		System.out.println("Estimating Flow SIZEs..." ); 
		Scanner sc = new Scanner(new File(filePath));
		System.out.println(filePath);
		String resultFilePath = GeneralUtil.path + "VSketch\\size\\v" + C[0].getDataStructureName()
				+ "_M_" +  M / 1024 / 1024 + "_u_" + u + "_m_" + m + "_T_" + times;;
		PrintWriter pw = new PrintWriter(new File(resultFilePath));
		System.out.println("Result directory: " + resultFilePath); 
		
		for (int t = 0; t < m; t++) {
			B[0] = B[0].join(C[0],w/m,t);
		}
		int totalSum1=B[0].getValue();
		int totalSum = C[0].getValue();
		if(C[0].getDataStructureName().equals("Bitmap")) {
			totalSum1=totalSum;
		}
	//	System.out.println("total spreads1: " + totalSum1);
		
		// Get the estimate of the physical data structure.
		
	//	System.out.println("total spreads: " + totalSum);
		int rr=0;
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			long flowid = GeneralUtil.getSize1FlowID(strs, true);
			int num = Integer.parseInt(strs[strs.length-1]);
				// Get the estimate of the virtual data structure.
				// encode segments, number of segments: w / m
				int virtualSum = C[0].getValueSegment(flowid, S, w / m);
				Double estimate = Math.max(1.0 * (virtualSum - 1.0 * m * totalSum1 / w), 1);
				Double threshold = 0.0;
				int sample_ratio = 16;
				if (estimate < threshold) {
					estimate=1.0;
				}
				pw.println(entry + "\t" + estimate.intValue());
				rr++;
		}
		sc.close();
		pw.close();
	}

	/** Encode elements to the physical data structure for flow spread measurement. */
	public static void encodeSpread(String filePath) throws FileNotFoundException {
		System.out.println("Encoding elements using " + C[0].getDataStructureName().toUpperCase() + "s for flow spread measurement..." );
		Scanner sc = new Scanner(new File(filePath));
		//n = 0;
		int nnn=0;
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			String[] res = GeneralUtil.getSperadFlowIDAndElementID(strs, true);
			long flowid = Long.parseLong(res[0]);
			long elementid = Long.parseLong(res[1]);
			C[0].encodeSegment(flowid, elementid, S, w / m);
		}
		System.out.println("Total number of encoded pakcets: " + n); 
		sc.close();
	}
	
	/** Estimate flow spreads. */
	public static void estimateSpread(String filePath) throws FileNotFoundException {
		System.out.println("Estimating Flow CARDINALITY..." ); 
		Scanner sc = new Scanner(new File(filePath));
		String resultFilePath = GeneralUtil.path + "VSketch\\spread\\v" + C[0].getDataStructureName()
				+ "_M_" +  M / 1024 / 1024 + "_u_" + u + "_m_" + m;
		PrintWriter pw = new PrintWriter(new File(resultFilePath));
		for (int t = 0; t < m; t++) {
			B[0] = B[0].join(C[0],w/m,t);
		}
		System.out.println("Result directory: " + resultFilePath); 
		int totalSum1=B[0].getValue();
		// Get the estimate of the physical data structure.
		int totalSum = C[0].getValue();
		if(C[0].getDataStructureName().equals("Bitmap")) {
			totalSum1=totalSum;
		}
	//	System.out.println("total spreads1: " + totalSum1);
	//	System.out.println("total spreads: " + totalSum);
		n=0;
		int rr=0;
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			long flowid = Long.parseLong(GeneralUtil.getSperadFlowIDAndElementID(strs, false)[0]);
			int num = Integer.parseInt(strs[strs.length-1]);
			n+=num;
				int virtualSum = C[0].getValueSegment(flowid, S, w / m);
				Double estimate = Math.max(1.0 * (virtualSum - 1.0 * m * totalSum1 / w), 1);

				rr++;
				int sample_ratio = 16;
				if (estimate < 0.0) {
					estimate=1.0;
				}
				pw.println(entry + "\t" + estimate.intValue());
		}
		sc.close();
		pw.close();
		// obtain estimation accuracy results
		System.out.println("total flow spread="+n);
	}
	
}
