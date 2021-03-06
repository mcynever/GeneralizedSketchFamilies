import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Scanner;
import java.util.stream.Collectors;

/** Some utility functions shared among general sketch families. */
public class GeneralUtil {	
	public static String path = "..\\..\\result\\";
	public static Boolean isDstAsID = true;
	public static HashSet<Integer> set1=new HashSet<Integer>(),set2=new HashSet<Integer>();
	/** parameter for estimation on a single source **/
	public static String dataStreamForFlowSize = "..\\..\\data\\datatrace.txt"; 	// Input for flow size measurement.
	public static String dataSummaryForFlowSize = "..\\..\\data\\srcdstsize.txt"; 	// Output for flow size measurement.
	public static String dataStreamForFlowSpread = "..\\..\\data\\datatrace.txt"; 	// Input for flow spread measurement.
	public static String dataSummaryForFlowSpread = "..\\..\\data\\dstspread.txt";	// Output for flow spread measurement.
	
	/** Get flow id for size measurement in each row of a file. */
	public static String getSizeFlowID(String[] strs, Boolean isEncoding) {
		if (strs.length == 0) return "";
		else if (isEncoding) return Arrays.stream(strs).collect(Collectors.joining("\t"));
		else return Arrays.stream(Arrays.copyOfRange(strs, 0, strs.length-1)).collect(Collectors.joining("\t"));
	}
	
	public static long getSize1FlowID(String[] strs, Boolean isEncoding) {
		if (strs.length == 0) return 0;
		else if (isEncoding) return ((Long.parseLong(strs[0])<<32)|Long.parseLong(strs[1]));
		else return Long.parseLong(strs[0]);
	}
	
	/** Get flow id and element id for spread measurement in each row of a file. */
	public static String[] getSpreadFlowIDAndElementID(String[] strs, Boolean isEncoding) {
		String[] res = new String[2];
		if (isEncoding) {
			if (isDstAsID) {res[0] = strs[1]; res[1] = strs[0];}
			else {res[0] = strs[0]; res[1] = strs[1];}
		} else {
			res[0] = strs[0];
		}
		return res;
	}
	
	/** A hash function that maps the string value to int value */
	public static int FNVHash1(String data) {
		final int p = 16777619;
		int hash = (int) 2166136261L;
		for (int i = 0; i < data.length(); i++)
			hash = (hash ^ data.charAt(i)) * p;
		hash += hash << 13;
		hash ^= hash >> 7;
		hash += hash << 3;
		hash ^= hash >> 17;
		hash += hash << 5;
		return hash;
	}
	
	/** FNVHash fucntion that takes two keys. */
	public static int FNVHash1(long key1, long key2) {
			  return FNVHash1((key1<<32) | key2);
	}
	
	/** FNVHash function **/
	public static int FNVHash1(long key) {
		  key = (~key) + (key << 18);
		  key = key ^ (key >>> 31);
		  key = key * 21; 
		  key = key ^ (key >>> 11);
		  key = key + (key << 6);
		  key = key ^ (key >>> 22);
		  return (int) key;
	}
	
	/** Thomas Wang hash */
	public static int intHash(int key) {
		key += ~(key << 15);
		key ^= (key >>> 10);
		key += (key << 3);
		key ^= (key >>> 6);
		key += ~(key << 11);
		key ^= (key >>> 16);
		return key;
	}
}

