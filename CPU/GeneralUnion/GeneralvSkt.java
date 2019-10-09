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
* A general framework for vSketch family. The elementary data structures to be plugged into can be counter, bitmap, FM sketch, HLL sketch. Specifically, we can
 * use counter to estimate flow sizes, and use bitmap, FM sketch and HLL sketch to estimate flow sizes/spreads.
 * @author Youlin
 */

public class GeneralvSkt {
	public static Random rand = new Random();
	
	public static int n = 0; 						// total number of packets
	public static int flows = 0; 					// total number of flows
	public static int avgAccess = 0; 				// average memory access for each packet
	public static int M = 1024 * 1024 * 8; 	// total memory space Mbits
	public static GeneralDataStructure[] C;
	public static GeneralDataStructure[] D;
	public static GeneralDataStructure[][] CP;
	public static Set<Integer> sizeMeasurementConfig = new HashSet<>(Arrays.asList(0)); // 0-couter; 1-Bitmap; 2-FM sketch; 3-HLL sketch
	public static Set<Integer> spreadMeasurementConfig = new HashSet<>(Arrays.asList()); // 1-Bitmap; 2-FM sketch; 3-HLL sketch
	public static boolean isGetThroughput = false; 
	
	/** parameters for sharing approach **/
	public static int u = 1;			// the unit size of each virtual data structure
	public static int w;				// number of the unit data structures in the physical array
	public static int m = 1;			// number of elementrary data structures in the virtual data structure
	public static int[] S; 				// random seeds for the sharing approach
	
	/** parameters for counters **/
	public static int mValueCounter = 128;
	public static int counterSize = 20;

	/** parameters for bitmap **/
	public static int bitmapSize = 1;	// sharing at bit level
	public static int virtualArrayLength = 20000;
	
	/** parameters for FM sketch **/
	public static int mValueFM = 128;
	public static int FMsketchSize = 32;
	
	/** parameters for hyperLogLog **/
	public static int mValueHLL = 128;
	public static int HLLSize = 5;
	
	public static int periods = 1;
	
	
	public static void main(String[] args) throws FileNotFoundException {
		System.out.println("Start************************");
		
		/** measurment for flow sizes **/
		for (int i : sizeMeasurementConfig) {
				initSharing(i);
				initJoining(i);
				encodeSize(GeneralUtil.dataStreamForFlowSize);
			  estimateSize(GeneralUtil.dataSummaryForFlowSize);
		}
		
		/** measurment for flow spreads **/
		for (int i : spreadMeasurementConfig) {
				initSharing(i);
				initJoining(i);
				encodeSpread(GeneralUtil.dataStreamForFlowSpread);
			  estimateSpread(GeneralUtil.dataSummaryForFlowSpread);
		}
		System.out.println("DONE!!!!!!!!!!!");
	}
	
	// Init the vSketch noise approach for different elementary data structures.
	public static void initJoining(int index) {
		switch (index) {
	        case 0:  D = new Counter[1]; D[0]=new Counter(mValueCounter,counterSize); 
	                 break;
	        case 1:  D = new Bitmap[1]; D[0] = new Bitmap(virtualArrayLength);
	                 break;
	        case 2:  D = new FMsketch[1]; D[0] = new FMsketch(mValueFM, FMsketchSize);
	                 break;
	        case 3:  D = new HyperLogLog[1]; D[0] = new HyperLogLog(mValueHLL, HLLSize);
	                 break;
	        default: break;
		}
		//generateSharingRandomSeeds();
		System.out.println("vSkt(" + C[0].getDataStructureName() + ") Initialized!");
	}
	// Init the vSketch approach for different elementary data structures.
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
		generatevSketchRamdonSeeds();
	}
	
	// Generate vSkt(counter) for flow size measurement.
	public static Counter[] generateCounter() {
		m = mValueCounter;
		u = counterSize;
		w = M / u;
		Counter[] B = new Counter[1];
		B[0] = new Counter(w, counterSize);
		
		Counter[][] BP = new Counter[periods][1];
		for(int t = 0; t < periods; t++) {
			BP[t][0] = new Counter(w, counterSize);
		}
		CP = BP;
		return B;
	}
		
	// Generate vSkt(bitmap) for flow size/spread measurement.
	public static Bitmap[] generateBitmap() {
		m = virtualArrayLength;
		u = bitmapSize;
		w = M / u;
		Bitmap[] B = new Bitmap[1];
		B[0] = new Bitmap(w);
		
		Bitmap[][] BP = new Bitmap[periods][1];
		for(int t = 0; t < periods; t++) {
			BP[t][0] = new Bitmap(w);
		}
		CP = BP;
		
		return B;
	}
	
	// Generate vSkt(FM) for flow size/spread measurement.
	public static FMsketch[] generateFMsketch() {
		m = mValueFM;
		u = FMsketchSize;
		w = M /u; 
		FMsketch[] B = new FMsketch[1];
		B[0] = new FMsketch(w, FMsketchSize);
		
		FMsketch[][] BP = new FMsketch[periods][1];
		for(int t = 0; t < periods; t++) {
			BP[t][0] = new FMsketch(w, FMsketchSize);
		}
		CP = BP;
		
		return B;
	}
	
	// Generate vSkt(HLL) for flow size/spread measurement.
	public static HyperLogLog[] generateHyperLogLog() {
		m = mValueHLL;
		u = HLLSize;
		w = M / u;
		HyperLogLog[] B = new HyperLogLog[1];
		B[0] = new HyperLogLog(w, HLLSize);
		
		HyperLogLog[][] BP = new HyperLogLog[periods][1];
		for(int t = 0; t < periods; t++) {
			BP[t][0] = new HyperLogLog(w, HLLSize);
		}
		CP = BP;
		
		return B;
	}
	
	// Generate random seeds for vSketch.
	public static void generatevSketchRamdonSeeds() {
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

	/** Encode elements to vSketch for flow size measurement. */
	public static void encodeSize(String filePath) throws FileNotFoundException {
		System.out.println("Encoding elements using " + C[0].getDataStructureName().toUpperCase() + "s..." );
		n = 0;
		for (int t = 0; t < periods; t++) {
			Scanner sc = new Scanner(new File(filePath + "splitIn"+periods+"\\output"+t+"v.txt"));
			System.out.println("Input file: " + filePath + "splitIn"+periods+"\\output"+t+"v.txt");
			while (sc.hasNextLine()) {
				String entry = sc.nextLine();
				String[] strs = entry.split("\\s+");
				long flowid=GeneralUtil.getSize1FlowID(strs, true);
		        n++;		
		        CP[t][0].encodeSegment(flowid, S, w / m);
			}
			sc.close();
			sc.close();
		}
		for (int t = 0; t < periods; t++) {
			C[0] = C[0].join(CP[t][0]);
		}
		System.out.println("Total number of encoded pakcets: " + n);
		
	}

	/** Estiamte flow sizes. */
	public static void estimateSize(String filePath) throws FileNotFoundException {
		System.out.println("Estimating Flow SIZEs..." ); 
		Scanner sc = new Scanner(new File(filePath));
		String resultFilePath = GeneralUtil.path + "VSketch\\size\\v" + C[0].getDataStructureName()
				+ "_M_" +  M / 1024 / 1024 + "_u_" + u + "_m_" + m + "_TS_" + periods;
		PrintWriter pw = new PrintWriter(new File(resultFilePath));
		System.out.println("Result directory: " + resultFilePath); 
		for (int t = 0; t < m; t++) {
			D[0] = D[0].join(C[0],w/m,t);
		}
		// Estimate noise.
		int totalSum = D[0].getValue();
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			long flowid=GeneralUtil.getSize1FlowID(strs, true);
			int num = Integer.parseInt(strs[strs.length-1]);
				// Get the estimate of the virtual data structure.
				int virtualSum = C[0].getValueSegment(flowid, S, w / m);
				Double estimate = Math.max(1.0 * (virtualSum - 1.0 * m * totalSum / w), 1);

				if (estimate < 0.0) {
					estimate = 1.0;
				}
				pw.println(entry + "\t" + estimate.intValue());
		}
		sc.close();
		pw.close();
	}

	/** Encode elements to the vSketch for flow spread measurement. */
	public static void encodeSpread(String filePath) throws FileNotFoundException {
		System.out.println("Encoding elements using " + C[0].getDataStructureName().toUpperCase() + "s..." );
		n = 0;
		for (int t = 0; t < periods; t++) {
			Scanner sc = new Scanner(new File(filePath + "splitIn"+periods+"\\output"+t+"v.txt"));

			System.out.println("Input file: " + filePath + "splitIn"+periods+"\\output"+t+"v.txt");
			while (sc.hasNextLine()) {
				String entry = sc.nextLine();
				String[] strs = entry.split("\\s+");
				String[] res = GeneralUtil.getSperadFlowIDAndElementID(strs, true);
				long flowid = Long.parseLong(res[0]);
				long elementid=Long.parseLong(res[1]);
				n++;
				CP[t][0].encodeSegment(flowid, elementid, S, w / m);
			}
			sc.close();
		}
		for (int t = 0; t < periods; t++) {
			C[0] = C[0].join(CP[t][0]);
		}
		System.out.println("Total number of encoded pakcets: " + n); 
		
	}
	
	/** Estiamte flow spreads. */
	public static void estimateSpread(String filePath) throws FileNotFoundException {
		System.out.println("Estimating Flow CARDINALITY..." ); 
		Scanner sc = new Scanner(new File(filePath));
		String resultFilePath = GeneralUtil.path + "VSketch\\spread\\v" + C[0].getDataStructureName()
				+ "_M_" +  M / 1024 / 1024 + "_u_" + u + "_m_" + m + "_TS_" + periods;;
		PrintWriter pw = new PrintWriter(new File(resultFilePath));
		System.out.println("Result directory: " + resultFilePath); 
		for (int t = 0; t < m; t++) {
			D[0] = D[0].join(C[0],w/m,t);
		}
		// Estimate noise.
		int totalSum = D[0].getValue();
		while (sc.hasNextLine()) {
			String entry = sc.nextLine();
			String[] strs = entry.split("\\s+");
			long flowid=Long.parseLong(GeneralUtil.getSperadFlowIDAndElementID(strs, false)[0]);
			int num = Integer.parseInt(strs[strs.length-1]);
				int virtualSum = C[0].getValueSegment(flowid, S, w / m);
				Double estimate = Math.max(1.0 * (virtualSum - 1.0 * m * totalSum / w), 1);

				if (estimate < 0.0) {
					estimate =1.0;
				}
				pw.println(entry + "\t" + estimate.intValue());
		}
		sc.close();
		pw.close();
	}

}
