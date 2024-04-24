import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;

import java.util.*;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataInputStream;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;


public class KMeans {

    // Centroids of k clusters.
    public static ArrayList<Double[]> centroids = new  ArrayList<>();
      
    public static void init_random_centroids(int k) throws IOException {
        // Generate a seed
        Random rand = new Random();
        
        // Generate k random centroids.
        for (int i = 0; i < k; i++) {

            Double[] centroid = new Double[2];

            centroid[0] = rand.nextDouble() * 100;
            centroid[1] = rand.nextDouble() * 100;

            centroids.add(centroid);
        }

        System.out.println("*******RANDOM CENTROIDS**********");
        for (int i= 0; i < centroids.size(); i++){
            System.out.println( "X: "+centroids.get(i)[0] + ", Y: " + centroids.get(i)[1]);
        }
    }

    private static void read_centroids(String centroid_path, int k) throws IOException {

        System.out.println("*******READ CENTROID FUNCTION**********");
        
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        // Path to centroids file in HDFS
        Path centroid_file_path = new Path("hdfs://localhost:9000"+centroid_path);

        if (fs.exists(centroid_file_path)) {

            // Clear all current centroids
            centroids.clear();

            try (FSDataInputStream input_stream = fs.open(centroid_file_path);
                InputStreamReader input_stream_reader = new InputStreamReader(input_stream);
                BufferedReader reader = new BufferedReader(input_stream_reader)) {
        
                String line;
                System.out.println("********FILE CONTENT**********");

                while ((line = reader.readLine()) != null) {
                    
                    System.out.println(line);
                    String[] parts = line.trim().split("\\s+");
                    Double[] centroid = new Double[2];

                    centroid[0] = Double.parseDouble(parts[0]);
                    centroid[1] = Double.parseDouble(parts[1]);
                    
                    centroids.add(centroid);
                }
            } catch (IOException e) {
                System.out.println("Exception: " + e.getMessage());
            }
        } else {
            System.out.println("File doesn't exist in HDFS.");
        }

        System.out.println("********FINISH READING CENTROIDS**********");
    }

    public static boolean has_converged(ArrayList<Double[]> old_centroids, ArrayList<Double[]> new_centroids, double threshold) {
        System.out.println("********CHECKING CONVERGENGE FUNCTION*********");

        // Check if centroids have converged based on a threshold
        boolean result = true;

        for (int i = 0; i < old_centroids.size(); i++) {

            Double[] old_centroid = old_centroids.get(i);
            Double[] new_centroid = new_centroids.get(i);
            double sum = 0.0;

            for (int j = 0; j < old_centroid.length; j++) {
                sum += Math.pow(old_centroid[j] - new_centroid[j], 2);
            }

            double distance = Math.sqrt(sum);

            System.out.println("Distance: " + distance);
            if (distance > threshold) {
                result = false;
            }
        }

        System.out.println("********CHECKED CONVERGENGE*********");
        if (result == true){
            System.out.println("Centroids have converged.");
        }else{
            System.out.println("Centroids are still changing.");
        }
        return result;
    }

    // Convert a centroids list to string
    public static String centroids_to_string()
    {
        StringBuilder centroids_string = new StringBuilder();

        for (Double[] centroid : centroids) {

            centroids_string.append(centroid[0]).append(",").append(centroid[1]).append(";");
        }
        
        return centroids_string.toString();
    }

    public static void rename_output_files(String output_dir) throws IOException
    {

        System.out.println("********RENAME OUTPUT FILE FUNCTION*********");

        FileSystem hdfs = FileSystem.get(new Configuration());
        Path output_path = new Path(output_dir);
        FileStatus fs[] = hdfs.listStatus(output_path);

        if (fs != null) {
            for (FileStatus file : fs) 
            {
                // If file is not a directory, remove the suffix
                if (!file.isDir()) {
                    String file_name = file.getPath().getName();
                    int index = file_name.indexOf("-r-00000");

                    if (index != -1) {
                        String new_file_name = file_name.substring(0, index);
                        Path new_path = new Path("hdfs://localhost:9000"+ output_dir + "/" + new_file_name);
                        
                        hdfs.rename(file.getPath(), new_path);
                    }
                }
            }
        }
    }

    // Mappper class
    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {

        public static ArrayList<Double[]> mapper_centroids = new  ArrayList<>();

        // Mehthod the make some setup before the Map phase.
        public void setup(Context context) throws IOException, InterruptedException {

            // Get centroids string from Kmeans class to Mapper class
            Configuration conf = context.getConfiguration();
            String centroids_string = conf.get("centroids");
        
            // Convert centroids_string to centroids_array
            String[] centroids_array = centroids_string.split(";");

            for (String centroid_str : centroids_array) {

                String[] parts = centroid_str.split(",");
                Double[] centroid = new Double[2];

                centroid[0] = Double.parseDouble(parts[0]);
                centroid[1] = Double.parseDouble(parts[1]);

                mapper_centroids.add(centroid);
            }
        }


        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            // Get the point from the value
            String line = value.toString();
            String[] point = line.split(" ");
            
            double x = Double.parseDouble(point[0]);
	        double y = Double.parseDouble(point[1]);
            double min_distance = Double.MAX_VALUE;
            int cluster = -1;

            for (int cluster_id = 0; cluster_id < mapper_centroids.size(); cluster_id ++) {

                double centroid_x = mapper_centroids.get(cluster_id)[0];
                double centroid_y = mapper_centroids.get(cluster_id)[1];
                double distance = Math.sqrt(Math.pow(centroid_x - x, 2) + Math.pow(centroid_y - y, 2));

                if (distance < min_distance) {
                    cluster = cluster_id;
                    min_distance = distance;
                }
            }

            // The key will be cluster id, value is the point string.
            context.write(new IntWritable(cluster), value);
        }
    }

    // Reducer class
    public static class KMeansReducer extends Reducer <IntWritable, Text, Text, IntWritable>{
 
        // A multiple output with key is Text type and value is IntWritable type.
        private MultipleOutputs<Text, IntWritable > mos;

        // Setup for the Multiple output.
        public void setup(Context context) throws IOException, InterruptedException {
            mos = new MultipleOutputs<>(context);
        }
    

        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Double sum_x = 0.0;
            Double sum_y = 0.0;
            int count = 1;

            // Sum the counts for the current cluster
            for (Text value : values){

                String line = value.toString();
                String[] point = line.split(" ");
                double x = Double.parseDouble(point[0]);
	            double y = Double.parseDouble(point[1]);

                sum_x += x;
                sum_y += y;
                count += 1;

                // Write cluster to classes file
                mos.write("cluster", value, key, "task_2_1.classes");
            }

            Double centroid_x = sum_x / count;
            Double centroid_y = sum_y / count;

            String centroid = centroid_x + " " + centroid_y;
	    
            mos.write("centroid", new Text(centroid), key, "task_2_1.clusters");
        }

        // Close the multiple output.
        public void cleanup(Context context) throws IOException, InterruptedException {
            mos.close();
        }

    }

    // Config a map reduce job
    public static void run_map_reduce_job(int k, String input_file, String output_file) throws Exception{

        Configuration conf = new Configuration();

        // Convert the centroid to string and set a new congiuration parameter.
        String centroids_string = centroids_to_string();
        conf.set("centroids", centroids_string);

        FileSystem fs = FileSystem.get(conf);
        Job job = Job.getInstance(conf, "KMeans Clustering");

        job.setJarByClass(KMeans.class);
        
        // Config Map phase
        job.setMapperClass(KMeansMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);	

        // Config Reduce phase.
        job.setReducerClass(KMeansReducer.class);
        job.setOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        
        // If output path has already existed, delete it before a job.
        Path output_path = new Path(output_file);
        if (fs.exists(output_path)) {
            fs.delete(output_path, true);
        }
	
        FileInputFormat.setInputPaths(job, new Path(input_file));
        FileOutputFormat.setOutputPath(job, output_path);
        MultipleOutputs.addNamedOutput(job, "centroid", TextOutputFormat.class,  Text.class, IntWritable.class);
        MultipleOutputs.addNamedOutput(job, "cluster", TextOutputFormat.class, Text.class, IntWritable.class);

        if (job.waitForCompletion(true)) {
            // If job completed, rename the output file. 
            rename_output_files(output_file);
        } else {
            System.out.println("MAP REDUCE JOB FAIL");
        }
    }

    // Run an interation
    public static void run(int iteration, int k, String input_file, String output_file) throws IOException {

        if (iteration == 0) {
            // Generate random centroids and write them to file
            init_random_centroids(k);
        }

        try{
            run_map_reduce_job(k, input_file, output_file);
            read_centroids(output_file + "/task_2_1.clusters", k);
            
        }catch (Exception e){
            System.out.println(e);
            System.out.println("ERROR");
        }
       
    }

    // args: 
    // - input_path
    // - output_path
    // - k
    // - max_iterations
    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.out.println("Invalid parameters. Use: <input_path> <output_path> <k> <max_iterations>");
            System.exit(1);
        }
        
        String input_file = args[0];
        String output_file = args[1];
        int k = Integer.parseInt(args[2]);
        int max_iterations = Integer.parseInt(args[3]);

		ArrayList<Double[]> old_centroids = new  ArrayList<>();
        ArrayList<Double[]> new_centroids = new  ArrayList<>();
        
		int iteration = 0;
  
		do {

            System.out.println("+++++++++++++++++");
            System.out.println("Iteration: " + iteration);

            if (iteration > 0){
                old_centroids = new  ArrayList<>(centroids);
            }
			run(iteration, k, input_file, output_file);

            if (iteration > 0){
                new_centroids = new  ArrayList<>(centroids);

                System.out.println("Old centroid");
                for (int i= 0; i < old_centroids.size(); i++){
                    System.out.println(old_centroids.get(i)[0] + " " + old_centroids.get(i)[1]);
                }

                System.out.println("New centroid");
                for (int i= 0; i < new_centroids.size(); i++){
                    System.out.println(new_centroids.get(i)[0] + " " + new_centroids.get(i)[1]);
                }

                if (has_converged(old_centroids, new_centroids,0.5) == true) {
                    break;
                }
            }

            iteration ++;
           
		} while ( iteration < max_iterations );

      }
    
}
