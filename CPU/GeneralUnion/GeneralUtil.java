import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.Collectors;

/** Util for general framework. */
public class GeneralUtil {	
	public static String path = "..\\..\\result\\";
	public static Boolean isDstAsID = true;
	public static String dataStreamForFlowSize = "..\\..\\data\\union\\"; 
	public static String dataSummaryForFlowSize = "..\\..\\data\\union\\srcdstsize.txt"; 
	public static String dataStreamForFlowSpread = "..\\..\\data\\union\\"; 
	public static String dataSummaryForFlowSpread = "..\\..\\data\\union\\dstspread.txt";
	
	/** get flow id for size measurment in each row of a file. */
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
	/** get flow id and element id for spread measurment in each row of a file. */
	public static String[] getSperadFlowIDAndElementID(String[] strs, Boolean isEncoding) {
		String[] res = new String[2];
		if (isEncoding) {
			if (isDstAsID) {res[0] = strs[1]; res[1] = strs[0];}
			else {res[0] = strs[0]; res[1] = strs[1];}
		} else {
			res[0] = strs[0];
		}
		return res;
	}

	
	/** a hash function that maps the string value to int value */
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
	
	public static int FNVHash1(long key1,long key2) {
		  //if(key1>(long)Integer.MAX_VALUE)
			//  return FNVHash1((key1-(1<<31))<<32+key2);
		  //else
			  return FNVHash1((key1<<32)|key2);
	}
	public static int FNVHash1(long key) {
		  key = (~key) + (key << 18); // key = (key << 18) - key - 1;
		  key = key ^ (key >>> 31);
		  key = key * 21; // key = (key + (key << 2)) + (key << 4);
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

