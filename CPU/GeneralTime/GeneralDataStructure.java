import java.util.BitSet;
import java.util.Random;

/** General data sturcture prototype for a unit data structure, such as counter, bitmap, etc. */
public abstract class GeneralDataStructure {
	// The random seed generator for this data structure.
	public static Random rand = new Random();
	// Default value for general setting.
	public final static int DEFAULT_CONSTANT = -1;
	
	
	public GeneralDataStructure() {
	}
	
	/** Get the name of the data structure. */
	public abstract String getDataStructureName();
	
	/** Get the size of the data structure. As an example of a bitmap, the value of u is the size of the bit array. */
	public abstract int getUnitSize();
	
	/** Get the value of the data structure. */
	public abstract int getValue();
	
	public abstract int[] getCounters();
	
	
	public abstract BitSet getBitmaps();
	
	public abstract BitSet[] getFMSketches();
	
	public abstract BitSet[] getHLLs();
	
	/** Encode an element in the data structure for size measurement. */
	public abstract void encode();
	
	/** Encode an element with elementID in the data structure for spread measurment. */
	public abstract void encode(String elementID);
	
	/** Encode an element with elementID in the flow with flowID to the shared data structure (in segments) for spread measurment. 
	 * @w number of segments
	 * */
	public abstract void encodeSegment(String flowID, String elementID, int[] s, int w);
	
	public abstract void encodeSegment(long flowID, long elementID, int[] s, int w);

	public abstract void encodeSegment(int flowID, int elementId, int[] s, int w);
	public abstract void encodeSegment(long flowID, int[] s, int w);
	
	/** Get the value of the flow with flowID in the shared data structure. */
	public int getValue(String flowID, int[] s) {
		return getValue();
	}
	public int getValue(long flowID, int[] s) {
		return getValue();
	}
	
	public abstract GeneralDataStructure join(GeneralDataStructure gds,int w,int i) ;
	public abstract GeneralDataStructure[] getsplit(int m,int w);
	/** Encode an element in the flow with flowID to the shared data structure (in segment) for size measurement. 
	 * @w number of segments
	 * */
	public abstract void encodeSegment(String flowID, int[] s, int w);
	
	public abstract void encodeSegment(int flowID, int[] s, int w);
	
	/** Get the value of the flow with flowID in the shared data structure (in segment). 
	 * @w number of segments
	 * */
	public int getValueSegment(String flowID, int[] s, int w) {
		return getValue();
	}
	public int getValueSegment(long flowID, int[] s, int w) {
		return getValue();
	}
	
	/** different encode function for throughput measurement*/
	public abstract void encode(int elementID);
	
	/** join operation among sketches **/
	public abstract GeneralDataStructure join(GeneralDataStructure gds);
}
