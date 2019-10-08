import java.util.BitSet;
import java.util.HashSet;

public class Bitmap extends GeneralDataStructure{
	/** parameters for bitmap */
	public int bitmapValue;										// the value derived from the bitmap
	public String name = "Bitmap";						// bitmap data structure name
	public int arrayLength;								// the size of the bit array
	
	public BitSet B;
	
	public Bitmap(int s) {
		super();
		arrayLength = s;
		B = new BitSet(s);
	}

	@Override
	public String getDataStructureName() {
		return name;
	}

	@Override
	public int getUnitSize() {
		return arrayLength;
	}
	
	@Override
	public BitSet getBitmaps() {
		return B;
	}
	
	@Override
	public int[] getCounters() {
		return null;
	};
	
	@Override
	public BitSet[] getFMSketches() {
		return null;
	};
	
	@Override
	public BitSet[] getHLLs() {
		return null;
	};	

	@Override
	public int getValue() {
		if (arrayLength == 1) {
			return B.cardinality();
		}
		return getValue(B);
	}
	
	@Override
	public void encode() {
		int k = rand.nextInt(arrayLength);
		B.set(k);
	}
	
	@Override
	public void encode(String elementID) {
		int k = (GeneralUtil.FNVHash1(elementID) % arrayLength + arrayLength) % arrayLength;
		B.set(k);
	}
	
	@Override
	public void encode(int elementID) {
		int k = (GeneralUtil.intHash(elementID) % arrayLength + arrayLength) % arrayLength;
		B.set(k);
	}
	
	
	@Override
	public void encodeSegment(String flowID, int[] s, int w) {
		int ms = s.length;
		int j = rand.nextInt(ms);
		int k = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % w + w) % w;
		int i = j * w + k;
		B.set(i);
	}
	
	@Override
	public void encodeSegment(long flowID, int[] s, int w) {
		int ms = s.length;
		int j = rand.nextInt(ms);
		int k = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % w + w) % w;
		int i = j * w + k;
		B.set(i);
	}
	
	@Override
	public void encodeSegment(int flowID, int[] s, int w) {
		int ms = s.length;
		int j = rand.nextInt(ms);			
		int k = (GeneralUtil.intHash(flowID ^ s[j]) % w + w) % w;
		int i = j * w + k;
		B.set(i);
	}
	
	@Override
	public GeneralDataStructure[] getsplit(int m,int w) {
		GeneralDataStructure[]  B=new Bitmap[m];
		for(int i=0;i<m;i++) {
			B[i]=new Bitmap(w);
			for(int j=0;j<w/m;j++)
				B[i].getBitmaps().set(j, this.getBitmaps().get(i*(w/m)+j));
		}
		return B;
	}
	@Override
	public void encodeSegment(long flowID, long elementID, int[] s, int w) {
		int m = s.length;
		int j = (GeneralUtil.FNVHash1(elementID^flowID) % m + m) % m;
		int k = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % w + w) % w;
		int i = j * w + k;
		B.set(i);
	}
	@Override
	public void encodeSegment(String flowID, String elementID, int[] s, int w) {
		int m = s.length;
		int j = (GeneralUtil.FNVHash1(elementID) % m + m) % m;
		int k = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % w + w) % w;
		int i = j * w + k;
		B.set(i);
	}
	
	@Override
	public void encodeSegment(int flowID, int elementID, int[] s, int w) {
		int m = s.length;
		int j = (GeneralUtil.intHash(elementID) % m + m) % m;
		int k = (GeneralUtil.intHash(flowID ^ s[j]) % w + w) % w;
		int i = j * w + k;
		B.set(i);
	}
	
	@Override
	public int getValue(String flowID, int[] s) {
		int ms = s.length;
		BitSet b = new BitSet(ms);
		for (int j = 0; j < ms; j++) {
			int i = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % arrayLength + arrayLength) % arrayLength;
			if (B.get(i)) b.set(j);
		}
		return getValue(b);
	}
	
	@Override
	public int getValue(long flowID, int[] s) {
		int ms = s.length;
		BitSet b = new BitSet(ms);
		for (int j = 0; j < ms; j++) {
			int i = (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % arrayLength + arrayLength) % arrayLength;
			if (B.get(i)) b.set(j);
		}
		return getValue(b);
	}
	
	@Override
	public int getValueSegment(String flowID, int[] s, int w) {
		int ms = s.length;
		BitSet b = new BitSet(ms);
		for (int j = 0; j < ms; j++) {
			int i = j * w + (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % w + w) % w;
			if (B.get(i)) b.set(j);
		}
		return getValue(b);
	}
	
	@Override
	public int getValueSegment(long flowID, int[] s, int w) {
		int ms = s.length;
		BitSet b = new BitSet(ms);
		for (int j = 0; j < ms; j++) {
			int i = j * w + (GeneralUtil.intHash(GeneralUtil.FNVHash1(flowID) ^ s[j]) % w + w) % w;
			if (B.get(i)) b.set(j);
		}
		return getValue(b);
	}
	
	
	public int getValue(BitSet b) {
		int zeros = 0;
		int len = b.size();
		for (int i = 0; i < len; i++) {
			if (!b.get(i)) zeros++;
		}
		Double result = -len * Math.log(1.0 * Math.max(zeros, 1)/ len);
		return result.intValue();
	}
	@Override
	public GeneralDataStructure join(GeneralDataStructure gds,int w,int i) {
		Bitmap b = (Bitmap) gds;
		for(int j=0;j<w;j++)
			if((b.getBitmaps().get(i*w+j))){
				B.set(i);
				break;
			}
		return this;
		
	}
	@Override
	public GeneralDataStructure join(GeneralDataStructure gds) {
		Bitmap b = (Bitmap) gds;
		B.or(b.B);
		return this;
	}
}
