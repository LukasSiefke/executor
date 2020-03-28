package opencl.executor;

import java.text.DecimalFormat;

public class Executor {

    public static class ExecutorFailureException extends Exception {
        public ExecutorFailureException(String message){
            super(message);
            System.err.println("Runtime error in the executor");
        }

        /// Consume the error. This restarts the Executor
        public void consume(){
            System.err.print("Restarting the executor... ");

            // Shutdown
            try { Executor.shutdown(); }
            catch( Exception e) { /* ignore shutdown errors */ }

            // restart
            try { Executor.init(); }
            catch( Exception e) {
                System.err.print("Catastrophic failure, cannot restart the executor");
                throw new RuntimeException("Cannot start Execution engine");
            }
            System.err.println("[ok]");
        }
    }

    public static KernelTime initAndExecute(Kernel kernel,
                                        int localSize1, int localSize2, int localSize3,
                                        int globalSize1, int globalSize2, int globalSize3,
                                        KernelArg[] args)
    {
        init();
        KernelTime runtime = execute(kernel, localSize1, localSize2, localSize3,
                globalSize1, globalSize2, globalSize3, args);
        shutdown();
        return runtime;
    }

    public static void loadLibrary()
    {
        System.loadLibrary("executor-jni");
    }

    public static void loadAndInit() {
        loadLibrary();
        init();
    }

    /** Compute matrix-matrix multiply natively */
    public static float[] nativeMatrixMultiply(float[] aa, float[] bb, int n, int m, int k) {
        float[] cc = new float[n*m];
        nativeMatrixMultiply(aa,bb,cc,n,m,k);
        return cc;
    }

    public native static void nativeMatrixMultiply(float[] a, float[] b, float[] out, int n, int m, int k);

    public native static KernelTime execute(Kernel kernel,
                                        int localSize1, int localSize2, int localSize3,
                                        int globalSize1, int globalSize2, int globalSize3,
                                        KernelArg[] args);

    /**
     * Executes the given kernel with the given runtime configuration and arguments iterations many
     * times. Returns the runtime of all runs in the order they were executed.
     * If the execution of a single run takes longer than the time specified in timeout only a
     * single iteration is performed. If the timeout is 0.0 this check will not be performed and
     * all iterations will be performed independent of the runtime of the kernel.
     *
     * @param kernel The kernel to execute
     * @param localSize1 Work-group size in dimension 0
     * @param localSize2 Work-group size in dimension 1
     * @param localSize3 Work-group size in dimension 2
     * @param globalSize1 Number of work-items in dimension 0
     * @param globalSize2 Number of work-items in dimension 1
     * @param globalSize3 Number of work-items in dimension 2
     * @param args Kernel arguments in the same order as expected by the kernel.
     * @param iterations The number of iterations to execute
     * @param timeOut A timeout specified in seconds. If the runtime of the kernel exceeds the
     *                timeout only a single iteration is executed. If the value is 0.0 this check
     *                will be disabled and all iterations will be performed independent of the
     *                kernel runtime.
     * @return An array of iterations many double values of runtimes measured using the OpenCL
     *         timing API. The ith value in the array is the runtime of the execution of the ith
     *         iteration.
     */
    public native static KernelTime[] benchmark(Kernel kernel,
                                            int localSize1, int localSize2, int localSize3,
                                            int globalSize1, int globalSize2, int globalSize3,
                                            KernelArg[] args, int iterations, double timeOut);

    public native static double evaluate(Kernel kernel,
                                         int localSize1, int localSize2, int localSize3,
                                         int globalSize1, int globalSize2, int globalSize3,
                                         KernelArg[] args, int iterations, double timeOut);

 

    public native static void init(int platformId, int deviceId);

    public native static long getDeviceLocalMemSize();

    public native static long getDeviceGlobalMemSize();

    public native static long getDeviceMaxMemAllocSize();

    public native static long getDeviceMaxWorkGroupSize();

    public native static boolean supportsDouble();

    public native static String getPlatformName();

    public native static String getDeviceName();

    public native static String getDeviceType();

    public static void init() {
        String platform = System.getenv("LIFT_PLATFORM");
        String device = System.getenv("LIFT_DEVICE");

        int platformId = 0;
        int deviceId = 0;

        if (platform != null) {
            try {
                platformId = Integer.parseInt(platform);
            } catch (NumberFormatException e) {
                System.err.println("Invalid platform id specified, using default.");
            }
        }

        if (device != null) {
            try {
                deviceId = Integer.parseInt(device);
            } catch (NumberFormatException e) {
                System.err.println("Invalid device id specified, using default.");
            }
        }

        init(platformId, deviceId);
    }

    public native static void shutdown();


    /**
    * Benchmark a OpenCL kernel.
    * 
    * @param kernelString     string containing the OpenCL kernelcode
    * @param kernelName       name of the kernel
    * @param buildOptions     options for the compiler
    * @param numberExecutions number of executions for the kernel
    * @param creator          KernelArgCreator for creating KernelArgs for the
    *                         kernel for every data size
    * @param dataSizesBytes   data sizes of the kernel arguments, which should be
    *                         tested, in bytes
    * @return result of benchmark-test
    */
    public static BenchmarkResult benchmark(String kernelString, String kernelName, String buildOptions,
        int numberExecutions, KernelArgCreator creator, int... dataSizesBytes) {

        if (dataSizesBytes == null)
            throw new NullPointerException();
        if (dataSizesBytes.length == 0)
            throw new IllegalArgumentException("not data sizes specificated");
        if (numberExecutions <= 0)
            throw new IllegalArgumentException("illegal number of executions: " + numberExecutions);

        // Absolute time Measurement
        long t0 = System.currentTimeMillis();

        // Create and compile Kernel
        Kernel kernel = Kernel.create(kernelString, kernelName, buildOptions);

        // Array for result
        KernelTime[][] result = new KernelTime[dataSizesBytes.length][numberExecutions];

        // Start run for every dataSize
        for (int i = 0; i < dataSizesBytes.length; i++) {
            int dataSize = dataSizesBytes[i];

            if (dataSize <= 0)
                throw new IllegalArgumentException();

            int dataLength = creator.getDataLength(dataSize);

			int local0 = creator.getLocal0(dataLength);
			int local1 = creator.getLocal1(dataLength);
			int local2 = creator.getLocal2(dataLength);
			int global0 = creator.getGlobal0(dataLength);
			int global1 = creator.getGlobal1(dataLength);
			int global2 = creator.getGlobal2(dataLength);

            // Create KernelArgs for this dataSize
            KernelArg[] args = creator.createArgs(dataLength);
            
            // Execute Kernel numberExecutions times
            result[i] = benchmark(kernel, local0, local1, local2, global0, global1, global2, args, numberExecutions, 0);
            
            // Destroy corresponding C-Objects
            for (KernelArg arg : args) {
                arg.dispose();
            }
        }

        // Absolute time Measurement
        long dt = System.currentTimeMillis() - t0;

        return new BenchmarkResult(numberExecutions, dataSizesBytes, result, kernelName, dt);
    }

    /**
    * Abstract class for generate KernelArgs with a specific size.
    */
    public static abstract class KernelArgCreator {
    /**
     * Returns the length of the data (number of elements).
     * 
     * @param dataSizeBytes size of data in bytes
     * @return length of the data
     */
    public abstract int getDataLength(int dataSizeBytes);

        /**
         * Generate KernelArgs.
         * 
         * @param dataLength length of the data (number of elements)
         * @return KernelArgs
         */
        public abstract KernelArg[] createArgs(int dataLength);

        public abstract int getLocal0(int dataLength);

        public int getLocal1(int dataLength) {
            return 1;
        }

        public int getLocal2(int dataLength) {
            return 1;
        }
        public abstract int getGlobal0(int dataLength);

        public int getGlobal1(int dataLength) {
            return 1;
        }

        public int getGlobal2(int dataLength) {
            return 1;
        }
    }

    /**
    * Class representing the result of a benchmark-test.
    */
    public static class BenchmarkResult {
        private final int numberExecutions;
        private final int[] dataSizes;
        private final KernelTime[][] result;
        private final KernelTime[] average;
        private final String kernelName;
        private final long testDuration;

        /**
         * Create a new result of benchmark-test.
         * 
         * @param executions   number of executions for the kernel for every data size
         * @param dataSizes    data sizes of the kernel arguments, which was tested, in
         *                     bytes
         * @param result       KernelTimes for every kernel execution for every datasize
         * @param kernelName   name of the tested kernel
         * @param testDuration duration of the test in milliseconds
         */
        protected BenchmarkResult(int numberExecutions, int[] dataSizes, KernelTime[][] result,
                String kernelName, long testDuration) {
            this.numberExecutions = numberExecutions;
            this.dataSizes = dataSizes;
            this.result = result;
            this.kernelName = kernelName;
            this.testDuration = testDuration;

            // Compute Average
            average = new KernelTime[result.length];
            for (int i = 0; i < dataSizes.length; i++) {
                double upload = 0;
                double download = 0;
                double launch = 0;
                double total = 0;

                for (int j = 0; j < numberExecutions; j++) {
                    upload += result[i][j].getUpload();
                    download += result[i][j].getDownload();
                    launch += result[i][j].getLaunch();
                    total += result[i][j].getTotal();
                }

                average[i] = new KernelTime((float) (upload / numberExecutions), (float) (download / numberExecutions),
                        (float) (launch / numberExecutions), (float) (total / numberExecutions));
            }
        }

        /**
         * Returns the number of executions for the kernel for every data size.
         * 
         * @return number of executions
         */
        public int getNumberExecutions() {
            return numberExecutions;
        }

        /**
         * Returns the data sizes of the kernel arguments, which was tested, in bytes.
         * 
         * @return data sizes, which was tested
         */
        public int[] getDataSizes() {
            return dataSizes;
        }

        /**
         * Returns the KernelTimes for every kernel execution for every datasize.
         * 
         * @return KernelTimes for kernel executions
         */
        public KernelTime[][] getResult() {
            return result;
        }

        /**
         * Returns the average KernelTimes for one kernel execution for every datasize.
         * 
         * @return average KernelTimes for one kernel execution for every datasize
         */
        public KernelTime[] getAverage() {
            return average;
        }

        /**
         * Returns the name of the tested kernel.
         * 
         * @return name of the tested kernel
         */
        public String getKernelName() {
            return kernelName;
        }

        @Override
        public String toString() {
            StringBuffer buffer = new StringBuffer(200);
            buffer.append("\nBenchmark " + kernelName + "-Kernel");

            buffer.append("  Datasize  Result (Average)\n");

            // For every dataSize: Append average for one kernel execution
            for (int i = 0; i < dataSizes.length; i++) {
                String dataSize = "" + humanReadableByteCountBin(dataSizes[i]);
                while (dataSize.length() < 10)
                    dataSize = " " + dataSize;

                String result = average[i].toString();

                buffer.append(dataSize);
                buffer.append(": ");
                buffer.append(result);
                buffer.append("\n");
            }

            // Absolute execution-time of the test
            DecimalFormat df = new DecimalFormat();
            String time = KernelTime.humanReadableMilliseconds(df, testDuration);
            df.setMaximumFractionDigits(1);
            String[] s = time.split(" ");

            if (s.length == 3)
                buffer.append("\nBenchmark-Duration: " + df.format(Double.parseDouble(s[0])) + " " + s[2] + "\n");
            else
                buffer.append("\nBenchmark-Duration: " + df.format(Double.parseDouble(s[0])) + " " + s[1] + "\n");

            return buffer.toString();
        }

        static String humanReadableByteCountBin(long bytes) {
            return bytes < 1024L ? bytes + " B"
                    : bytes <= 0xfffccccccccccccL >> 40 ? String.format("%.1f KiB", bytes / 0x1p10)
                            : bytes <= 0xfffccccccccccccL >> 30 ? String.format("%.1f MiB", bytes / 0x1p20)
                                    : String.format("%.1f GiB", bytes / 0x1p30);
        }
    }
}
