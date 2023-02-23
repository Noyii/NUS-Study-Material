/*
ENTER YOUR NAME HERE
NAME: NIHARIKA SHRIVASTAVA
MATRICULATION NUMBER: A0254355A
*/

import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class TopkCommonWords {

  // Wrapper for ArrayWritable so that Reducer can take it as an input value
  public static class TextArrayWritable extends ArrayWritable {
    public TextArrayWritable() {
        super(Text.class);
    }

    public TextArrayWritable(String[] strings) {
        super(Text.class);
        Text[] texts = new Text[strings.length];
        for (int i = 0; i < strings.length; i++) {
            texts[i] = new Text(strings[i]);
        }
        set(texts);
    }
  }


  // Mapper
  public static class CounterMapper extends Mapper<Object, Text, Text, ArrayWritable> {
    // Using a generic key for all words in order to have only 1 reducer. 
    // Not efficient but helps to collate results between different mappers.
    private final static Text genericKey = new Text("key");

    // Stores count of word in each file
    private HashMap<String, Integer> wordMap = new HashMap<String, Integer>();

    // Initialize stopwords only once
    private List<String> stopwords = null;

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      if (stopwords == null) {
        Configuration conf = context.getConfiguration();
        stopwords = Arrays.asList(conf.get("STOPWORDS").split("\n"));  
      }
      StringTokenizer itr = new StringTokenizer(value.toString());

      while (itr.hasMoreTokens()) {
        String word = itr.nextToken();

        // process only those words which have length greater than 4 and are not stopwords
        if (word.length() > 4 && !stopwords.contains(word)) {

          // Put entry in hashmap
          if (wordMap.containsKey(word)) {
            wordMap.put(word, wordMap.get(word)+1);
          }
          else {
            wordMap.put(word, 1);
          }
        }
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();

      for (HashMap.Entry<String, Integer> set : wordMap.entrySet()) {
        String[] pair = new String[]{fileName, set.getKey(), set.getValue().toString()};

        // Emit {"key" :[word, count]} for the entire chunk
        context.write(genericKey, new TextArrayWritable(pair));
      }
      super.cleanup(context);
    }
  }


  // Reducer
  public static class TopKReducer extends Reducer<Text, TextArrayWritable, IntWritable, Text> {
    private Map<String, Integer> wordMap = new HashMap<String, Integer>();
    private HashMap<String, HashMap<String, Integer>> fileCounts = new HashMap<String, HashMap<String, Integer>>();

    public void reduce(Text key, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
      // Get the value K
      Configuration conf = context.getConfiguration();
      int K = Integer.parseInt(conf.get("K"));

      for (TextArrayWritable val : values) {
        String fileName = (val.get()[0]).toString();
        String word = (val.get()[1]).toString();
        int count = Integer.parseInt((val.get()[2]).toString());

        if (fileCounts.containsKey(fileName)) {
          HashMap<String, Integer> wordCount = fileCounts.get(fileName);

          if (wordCount.containsKey(word)) {
            wordCount.put(word, wordCount.get(word) + count);
          }
          else {
            wordCount.put(word, count);
          }

          fileCounts.put(fileName, wordCount);
        }
        else {
          HashMap<String, Integer> wordCount = new HashMap<String, Integer>();
          wordCount.put(word, count);
          fileCounts.put(fileName, wordCount);
        }
      }
      
      Set<String> keys = fileCounts.keySet();
      String[] files = keys.toArray(new String[keys.size()]);
      HashMap<String, Integer> File1 = fileCounts.get(files[0]);
      HashMap<String, Integer> File2 = fileCounts.get(files[1]);

      // wordMap has the min count from the 2 files
      for (HashMap.Entry<String, Integer> set : File1.entrySet()) {
        String word = set.getKey();
        if (File2.containsKey(word)) {
          wordMap.put(word, Math.min(set.getValue(), File2.get(word)));
        }
      }

      // Sort by value in descending order and then by lexicographically in case frequency is same
      Map<String, Integer> sortedWordMap = sortByValue(wordMap);
      int i = 0;

      // Output the top K entries from the sorted map
      for (Map.Entry<String, Integer> set : sortedWordMap.entrySet()) {
        if (i >= K) {break;}
        
        IntWritable count = new IntWritable(set.getValue());
        Text word = new Text(set.getKey());
        context.write(count, word);
        i += 1;
      }
    }

    // Custom sorting function 
    private static Map<String, Integer> sortByValue(Map<String, Integer> unsortMap) {
        List<Entry<String, Integer>> list = new LinkedList<>(unsortMap.entrySet());

        list.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()) != 0 ? 
                                o2.getValue().compareTo(o1.getValue()) : // Sort in descending order using values
                                  o1.getKey().compareTo(o2.getKey()));   // Sort lexicographically when frequency is same

        return list.stream().collect(Collectors.toMap(Entry::getKey, Entry::getValue, (a, b) -> b, LinkedHashMap::new));
    }
  }


  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    
    // Set input arguments as parameters so that it can be used in mapper/reducer
    String stopwords = Files.readString(java.nio.file.Path.of(args[2]));
    conf.set("STOPWORDS", stopwords);
    conf.set("K", args[4]);
    
    // Set job configurations
    Job job = Job.getInstance(conf, "top k common words");
    job.setJarByClass(TopkCommonWords.class);
    job.setMapperClass(CounterMapper.class);
    job.setReducerClass(TopKReducer.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(TextArrayWritable.class);

    // Set I/O paths 
    String paths = args[0] + "," + args[1];
    FileInputFormat.addInputPaths(job, paths);
    FileOutputFormat.setOutputPath(job, new Path(args[3]));

    System.exit(job.waitForCompletion(true) ? 0 : 1);
  } 
}
