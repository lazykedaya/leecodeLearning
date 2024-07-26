
import java.awt.geom.Ellipse2D;
import java.awt.geom.RoundRectangle2D;
import java.io.OutputStream;
import java.sql.SQLOutput;
import java.time.chrono.IsoChronology;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class array_test {
    public static void main(String[] args) {


        String[] nums2 = {"Science", "is", "what", "we", "understand", "well",
                "enough", "to", "explain", "to", "a", "computer.", "Art",
                "is", "everything", "else", "we", "do"};
        int[][] nums1 = {{1, 2, 3}, {5, 6, 7}, {9, 10, 11}};
        int[] nums3 = {3, 1, 2};
        System.out.println(findMin(nums3));
    }

    //XXXXXX寻找两个正序数组的中位数
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        //思路1，把两个数组合并，然后输出中间索引的数，显然时间复杂度较高
        //思路2，
        return 1.0;
    }

    //寻找旋转排序数组中的最小值
    public static int findMin(int[] nums) {
        int n = nums.length;
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = (right + left) / 2;
            if (nums[mid] < nums[right]) right = mid;
            else left = mid + 1;
        }
        return nums[left];
    }

    //在排序数组中查找元素的第一个和最后一个位置
    public static int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) return new int[]{-1, -1};
        int n = nums.length;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = (right + left) / 2;
            if (nums[mid] == target) {
                left = mid;
                right = mid;
                while (left - 1 >= 0 && nums[left - 1] == target) {
                    left--;
                }
                while (right + 1 < n && nums[right + 1] == target) {
                    right++;
                }
                return new int[]{left, right};
            }
            if (nums[mid] < target) {
                left = mid + 1;
            } else right = mid - 1;
        }
        return new int[]{-1, -1};
    }

    //搜索旋转排序数组
    public static int search(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            if (nums[left] == target) return left;
            if (nums[right] == target) return right;
            if (nums[mid] > nums[left] && nums[mid] > nums[right]) {
                if (target > nums[mid]) {
                    left = mid + 1;
                } else if (target < nums[right]) left = mid + 1;
                else if (target > nums[right]) right = mid - 1;
                continue;
            }
            if (nums[mid] < nums[left] && nums[mid] < nums[right]) {
                if (target < nums[mid]) {
                    right = mid - 1;
                } else if (target < nums[right]) left = mid + 1;
                else if (target > nums[right]) right = mid - 1;
                continue;
            }

            if (target > nums[mid]) {
                left = mid + 1;
            } else right = mid - 1;
        }
        return -1;
    }

    //寻找峰值
    public static int findPeakElement(int[] nums) {
        //一般思路遍历，对每个前后进行比较，找到即返回
        int n = nums.length;
        int left = 0, right = n - 1, ans = -1;
        while (left <= right) {
            int mid = (right + left) / 2;
            if (compare(nums, mid - 1, mid) < 0 && compare(nums, mid, mid + 1) > 0) {
                ans = mid;
                break;
            }
            if (compare(nums, mid, mid + 1) < 0) {
                left = mid + 1;
            } else right = mid - 1;
        }
        return ans;
    }

    private static int compare(int[] nums, int idx1, int idx2) {
        int[] num1 = get(nums, idx1);
        int[] num2 = get(nums, idx2);
        if (num2[0] != num1[0]) {
            return num1[0] > num2[0] ? 1 : -1;
        }
        if (num2[1] == num1[1]) {
            return 0;
        }
        return num1[1] > num2[1] ? 1 : -1;

    }

    private static int[] get(int[] nums, int idx1) {
        if (idx1 == -1 || idx1 == nums.length) {
            return new int[]{0, 0};
        }
        return new int[]{1, nums[idx1]};
    }

    //搜索二维矩阵
    public static boolean searchMatrix(int[][] matrix, int target) {
        //可以先判断在哪行再判断在哪列，也可以直接把右指针定为m*n-1；判断时需要把指针数转为二维数组坐标i=right/n,j=right%n
        if (matrix == null) return false;
        int m = matrix.length, n = matrix[0].length;
        if (matrix[0][0] > target || matrix[m - 1][n - 1] < target) return false;
        int low = 0, high = m - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (matrix[mid][n - 1] == target) return true;
            if (matrix[mid][n - 1] > target) {
                high = mid - 1;
            } else low = mid + 1;
        }
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (matrix[low][mid] == target) return true;
            if (matrix[low][mid] > target) {
                right = mid - 1;
            } else left = mid + 1;
        }
        return false;
    }

    //文本左右对齐
    public static List<String> fullJustify(String[] words, int maxWidth) {
        //代码有待优化
        /*int len = words.length;
        ArrayList<String> strList = new ArrayList<>();
        int total = words[0].length();
        int old = 0, neW = 1;
        while (true) {
            if (neW == len) {
                StringBuffer strL = new StringBuffer();
                for (int i = old; i < len - 1; i++) {
                    strL.append(words[i] + " ");
                }
                strL.append(words[len - 1]);
                for (int i = strL.length(); i < 16; i++) {
                    strL.append(" ");
                }
                System.out.println(strL.length());
                strList.add(strL.toString());
                return strList;
            }
            total += words[neW].length();
            if ((total + neW - old) < 16) {
                neW++;
            } else {
                //大于16的时候，把old-neW之间的单词组合放一行,同时把toal归零，old换成neW，neW++；
                //re表示一行的单词个数
                int re = neW - old;
                if (re >= 2) {
                    //ko表示该行不差空情况下多出的位置
                    int ko = 16 - (total - words[neW].length());
                    //平均每两个单词间加入多少个空
                    int ko1 = ko / (re - 1);
                    //前面几个多一个
                    int Ko2 = ko % (re - 1);
                    //得到“ko1空格”
                    StringBuffer stepK = new StringBuffer();
                    for (int i = 0; i < ko1; i++) {
                        stepK.append(" ");
                    }
                    //添加到strList中
                    StringBuffer str = new StringBuffer();
                    while (old < neW - 1) {
                        for (int i = 0; i < re - 1; i++) {
                            if (i < Ko2) {
                                str.append(words[old] + stepK + " ");
                                old++;
                            } else {
                                str.append(words[old] + stepK);
                                old++;
                            }
                        }
                    }
                    str.append(words[old++]);
                    System.out.println(str.length());
                    strList.add(str.toString());
                }else {
                    StringBuffer strL3 = new StringBuffer();
                    strL3.append(words[old]);
                    for (int i = strL3.length(); i < 16; i++) {
                        strL3.append(" ");
                    }
                    System.out.println(strL3.length());
                    strList.add(strL3.toString());

                }
                total = words[neW].length();
                neW++;
            }
        }*/
        //优化后
        List<String> res = new ArrayList<>();
        int right = 0, n = words.length;
        while (right < words.length) {
            int numWord = 0, sumLen = 0, start = right;
            while (true) {
                sumLen += words[right].length();
                numWord++;
                right++;
                if (right >= n || sumLen + words[right].length() + numWord > maxWidth) break;
            }
            //如果是最后一行
            if (right == n) {
                StringBuilder sb = new StringBuilder();
                for (int j = start; j < n - 1; j++) {
                    sb.append(words[j]);
                    sb.append(' ');
                }
                sb.append(words[n - 1]);
                int size = sb.length();
                for (int j = size; j < maxWidth; j++) sb.append(' ');
                res.add(sb.toString());
                break;
            }
            //如果该行只有一个单词
            if (numWord == 1) {
                StringBuilder sb = new StringBuilder();
                sb.append(words[start]);
                int size = sb.length();
                for (int j = size; j < maxWidth; j++) sb.append(' ');
                res.add(sb.toString());
                continue;
            }
            int numBlank = maxWidth - sumLen;
            StringBuilder sb = new StringBuilder();
            //如果是中间行并且超过一个单词
            for (int j = start; j < right - 1; j++) {
                sb.append(words[j]);
                for (int k = 0; k < numBlank / (numWord - 1); k++) {
                    sb.append(' ');
                }
                if (j - start < numBlank % (numWord - 1)) sb.append(' ');
            }
            sb.append(words[right - 1]);
            res.add(sb.toString());
        }
        return res;
    }

    //44Z字形变换***
    public static String convert(String s, int numRows) {
        //找规律（用矩阵模拟填入）
        //int t = r * 2 - 2;
        //int c = (n + t - 1) / t * (r - 1);
        //char[][] mat = new char[r][c];
        int len = s.length();
        if (numRows == 1) return s;
        List<StringBuffer> lists = new ArrayList<>();
        //一次循环个数
        int circle = 2 * numRows - 2;
        //用来i%circle来确定放哪行
        int[] index = new int[circle];
        for (int i = 0; i < circle; i++) {
            if (i < numRows) {
                index[i] = i;
            } else {
                index[i] = circle - i;
            }
        }
        //每行建一个stringbuffer
        for (int i = 0; i < numRows; i++) {
            lists.add(i, new StringBuffer());
        }
        //s循环，一个一个判断加到哪一行并合并
        for (int i = 0; i < len; i++) {
            int indexI = i % circle;
            lists.get(index[indexI]).append(s.charAt(i));
        }
        //合并每一行
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < numRows; i++) {
            ans.append(lists.get(i));
        }
        return ans.toString();


    }

    //xxxxxxx串联所有单词的子串(words 中所有字符串 长度相同)
    public static List<Integer> findSubstring(String s, String[] words) {
        return new ArrayList<>();
    }

    //*****旋转图像
    public static void rotate(int[][] matrix) {
        //1行变列，用新矩阵存着
       /* int len=matrix.length;
        int [][]matrix0=new int[len][len];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                matrix0[j][len-i-1]=matrix[i][j];
            }
        }
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                matrix[i][j]=matrix0[i][j];
            }
        }*/
        //2分成四个小区域，保证不覆盖
        /*int len=matrix.length;
        for (int i = 0; i < len / 2; i++) {
            for (int j = 0; j < (len + 1) / 2; j++) {
                int temp=matrix[i][j];
                matrix[i][j]=matrix[len-j-1][i];
                matrix[len-j-1][i]=matrix[len-i-1][len-j-1];
                matrix[len-i-1][len-j-1]=matrix[j][len-i-1];
                matrix[j][len-i-1]=temp;
            }
        }*/
        //用翻转代替旋转,先水平再主对角线
        int len = matrix.length;
        for (int i = 0; i < len / 2; i++) {
            for (int j = 0; j < len; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[len - i - 1][j];
                matrix[len - i - 1][j] = temp;
            }
        }
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }


    }

    //螺旋矩阵
    public static List<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> ansList = new ArrayList<>();
        ansList.add(matrix[0][0]);
        //四个方向//si个行列指针
        int row = matrix.length;
        int col = matrix[0].length;
        int row0 = 0;
        int col0 = 0;
        //对应顺时针方向()右下左上。。。
        int dx[] = {0, 1, 0, -1};
        int dy[] = {1, 0, -1, 0};
        //当前坐标
        int px = 0, py = 0;
        //初始方向
        int index = 0;
        int count = 0;
        int num = col * row;
        while (count < (num - 1)) {
            //先保持上一次的方向
            int nextPx = px + dx[index];//行
            int nextPy = py + dy[index];//列
            if (nextPy >= col) {
                //如果到右上头 小行+1下走
                row0++;
                index = 1;
            } else if (nextPy < col0) {
                //如果到左下头 大行-1上走
                row--;
                index = 3;
            } else if (nextPx >= row) {
                //到右下头 大列-1，左走
                col--;
                index = 2;
            } else if (nextPx < row0) {
                //到左上头 小列+1，右走
                col0++;
                index = 0;
            }
            px = px + dx[index];
            py = py + dy[index];
            ansList.add(matrix[px][py]);
            count++;
        }
        return ansList;
    }

    //接雨水
    public static int trap(int[] height) {
       /* 超时了
       //(分层+滑块）分多少层？一层代表一个循环，显然不合适
        //滑块，两个循环
        int left = 0, right = 1;
        int sum = 0;
        int len = height.length;
        if (len < 3) return 0;
        while (left < len - 1) {
            //除去开头的空格
            while (height[left] == 0 && right < len) {
                left++;
                right++;
            }//还没有考虑right出界的问题
            // 找到>=left的地方，并且right-left要大于1
            int max = 0;
            int max_index = left;
            while (right < len && height[right] < height[left]) {
                //记住最大值方便没找到时更新left；
                if (height[right] >= max) {
                    max = height[right];
                    max_index = right;
                }
                right++;
            }
            //如果没找到，表明left点是最高点，后续最高点是max,如果max点-left点>1的话也可以
            if (right == len) {
                if ((max_index - left) > 1) {
                    for (int i = left + 1; i < max_index; i++) {
                        sum += (height[max_index] - height[i]);
                    }
                }
                //更新
                left = right;
                right++;
                continue;
            }
            //如果找到了；并且right-left大于1，两个不相互连
            if ((right - left) > 1) {
                for (int i = left + 1; i < right; i++) {
                    sum += (height[left] - height[i]);
                }
            }
            //更新left和right位置
            left = right;
            right++;
        }
        return sum;*/
        //1,双指针
        /*int sum=0;
        int left=0,right=height.length-1;
        int leftMax=0,rightMax=0;
        while (left<right){
            leftMax=Math.max(leftMax,height[left]);
            rightMax=Math.max(rightMax,height[right]);
            if(height[left]<height[right]){
                sum+=leftMax-height[left];
                ++left;
            }else {
                sum+=rightMax-height[right];
                --right;
            }
        }
        return sum;*/
        //2,动态规划，正反取最小
        int len = height.length;
        int left_max = height[0];
        int[] ints = new int[len];
        int sum = 0;
        for (int i = 0; i < len; i++) {
            if (left_max >= height[i]) {
                ints[i] = left_max - height[i];
            } else {
                left_max = height[i];
                ints[i] = 0;
            }
        }
        int right_max = height[len - 1];
        for (int i = len - 1; i >= 0; i--) {
            if (right_max >= height[i]) {
                ints[i] = Math.min(ints[i], right_max - height[i]);
            } else {
                right_max = height[i];
                ints[i] = 0;
            }
            sum += ints[i];
        }
        return sum;
        //3.
    }

    //分发糖果
    public static int candy(int[] ratings) {
        //找到局部极小点和极大点；或者第一个数从1开始，后面根据最小（n）的情况，有负数全体加（-n+1）
        int len = ratings.length;
        int[] left = new int[len];
        //满足左规则
        for (int i = 0; i < len; i++) {
            if (i > 0 && ratings[i] > ratings[i - 1]) {
                left[i] = left[i - 1] + 1;
            } else {
                left[i] = 1;
            }
        }
        int right = 0, ret = 0;
        for (int i = len - 1; i >= 0; i--) {
            if (i < len - 1 && ratings[i] > ratings[i + 1]) {
                right++;
            } else right = 1;
            ret += Math.max(right, left[i]);
        }
        return ret;

    }

    //栈-简化路径
    public static String simplifyPath(String path) {
        String[] split = path.split("/");
        ArrayList<String> list = new ArrayList<>();
        int index = 1;
        for (int i = 0; i < split.length; i++) {
            String str = split[i];
            if (!str.isEmpty()) {
                if (str.equals(".")) continue;
                else if (str.equals("..") && !list.isEmpty()) {
                    index--;
                    list.remove(index - 1);
                } else {
                    if (split[i].equals(".."))
                        continue;
                    list.add("/" + split[i]);
                    index++;
                }
            }
        }

        StringBuffer ans = new StringBuffer();
        if (list.isEmpty()) return "/";
        for (String s : list) {
            ans.append(s);
        }
        return ans.toString();


        //先拆分，ArrayDeque是 Deque的实现类，可以作为栈来使用，效率高于 Stack；也可以作为队列来使用，效率高于 LinkedList。
        //ArrayDeque 是 Java 集合中双端队列的数组实现，双端队列的链表实现（LinkedList），可以前后添加，前后删除等

       /* Stack<String> stack = new Stack<>();
        stack.push("/");
        for (int i = 1; i < split.length; i++) {
            String a = split[i];
            if (a.isEmpty() && stack.peek() != "/") stack.push("/");
            else if (a.equals("..") && !stack.isEmpty()) stack.pop();
            else if (!a.equals(".")) stack.push(a);
        }
        if (stack.isEmpty()) return "/";
        return stack.toString();*/
        /*StringBuffer res = new StringBuffer();*/

    }

    //栈-有效的括号
    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char a = s.charAt(i);
            if (a == '(' || a == '[' || a == '{') {
                stack.push(a);
            } else {
                //判断抵消左右括号
                if (stack.isEmpty()) return false;
                //peek只查看不移出
                if (a == ')' && stack.peek() != '(') return false;
                if (a == ']' && stack.peek() != '[') return false;
                if (a == '}' && stack.peek() != '{') return false;
                //pop会移出栈顶内容
                stack.pop();
            }
        }
        //判断是否有多余的左括号
        return stack.isEmpty();
    }

    //有效的数独(矩阵)
    public static boolean isValidSudoku(char[][] board) {
        //暴力解法，对每一行，每一列进行验证，用3个哈希表保存
        int[][] rows = new int[9][9];
        int[][] cols = new int[9][9];
        int[][][] box = new int[3][3][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c != '.') {
                    int index = c - '0' - 1;
                    rows[i][index]++;
                    cols[index][j]++;
                    box[i / 3][j / 3][index]++;
                    if (rows[i][index] > 1 || cols[index][j] > 1 || box[i / 3][j / 3][index] > 1) return false;
                }
            }
        }
        return true;
    }

    //长度最小的子数组
    public static int minSubArrayLen(int target, int[] nums) {
        //复杂度较高
        int len = nums.length;
        if (len == 0) return 0;
        /*int min=len+1;
        for (int i = 0; i < len; i++) {
            int sum=0;
            for (int j = i; j < len; j++) {
                sum+=nums[j];
                if(sum>=target){
                    min=Math.min((j-i+1),min);
                    break;
                }
                if(j==len-1)return 0;
            }
        }
        return min;*/
        //使用滑动窗口
        int start = 0, end = 0;
        int ans = len + 1;
        int sum = 0;
        while (end < len) {
            sum += nums[end];
            while (sum >= target) {
                ans = Math.min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == len + 1 ? 0 : ans;

    }

    //直线上最多的点
    public static int maxPoints(int[][] points) {
        int len = points.length;
        if (len <= 2) return len;
        //枚举，用两个循环，
        int ret = 0;
        for (int i = 0; i < len; i++) {
            if (ret >= len - 1 || ret > len / 2) break;
            HashMap<Integer, Integer> map = new HashMap<>();
            for (int j = i + 1; j < len; j++) {
                //i点后面每一个点与其斜率，斜率不能用除法求，用map保留x,y
                int x = points[i][0] - points[j][0];
                int y = points[i][1] - points[j][1];
                if (x == 0) y = 1;
                else if (y == 0) x = 1;
                else {
                    if (y < 0) {
                        x = -x;
                        y = -y;
                    }
                    //约分，先求最小公因子

                    int gcdXY = gcd(Math.abs(x), Math.abs(y));
                    x /= gcdXY;
                    y /= gcdXY;
                }
                int key = y + x * 200001;
                map.put(key, map.getOrDefault(key, 0) + 1);
            }
            int max = 0;
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                int num = entry.getValue();
                max = Math.max(max, num + 1);
            }
            ret = Math.max(ret, max);
        }
        return ret;
    }

    public static int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a;
    }

    //三角形最小路径和（动态规划***）
    public static int minimumTotal(List<List<Integer>> triangle) {
        int len = triangle.size();
        int[][] f = new int[len][len];
        f[0][0] = triangle.get(0).get(0);
        for (int i = 1; i < len; i++) {
            f[i][0] = f[i - 1][0] + triangle.get(i).get(0);
            for (int j = 1; j < i; j++) {
                f[i][j] = Math.min(f[i - 1][j - 1], f[i - 1][j]) + triangle.get(i).get(j);
            }
            f[i][i] = f[i - 1][i - 1] + triangle.get(i).get(i);
        }
        //通过这步使得f最后一行为triangle最后一行的每一个数往上爬的最小路径，最后比较一下得到全局最小
        int min = f[len - 1][0];
        for (int i = 1; i < len; i++) {
            min = Math.min(f[len - 1][i], min);
        }
        return min;
        //进一步优化就是，从下往上走；
        /*int n = triangle.size();
        int[] dp = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];*/

    }

    //阶乘后的零
    public static int trailingZeroes(int n) {

        int num5 = 0;
        if (n < 5) return 0;
        //关键在于5和0的数量不看2的数量
        for (int i = 5; i <= n; i += 5) {
            int temp = i;
            while (temp % 5 == 0) {
                num5++;
                temp = temp / 5;
            }
        }
        return num5;

        //优化计算,n里面含5的倍数的有n/5个，含25的倍数的个数有n/25个。。。
        /*int ans = 0;
        while (n) {
            n /= 5;
            ans += n;
        }
        return ans;*/

    }

    //50回文数
    public static boolean isPalindrome(int x) {
        //如果要追求空间小可以用翻转一半的方法
        String s = x + "";
        char[] chars = s.toCharArray();
        int i = 0, j = chars.length - 1;
        while (i < j) {
            if (chars[i] != chars[j]) return false;
            i++;
            j--;
        }
        return true;
    }

    //Pow(x, n)
    public static double myPow(double x, int n) {
        //用的时间较长
        /*if(n==0||x==1)return 1;
        if(x==-1)return n%2==0?1:-1;
        double ans=1;int me=Math.abs(n);
        for (double i = 0; i < me; i++) {
            ans*=x;
        }
        if(n>0) return ans;
        return (1.0/ans);*/
        //快速幂
        long N = n;
        return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);

    }

    public static double quickMul(double x, long N) {
        double ans = 1.0;
        // 贡献的初始值为 x
        double x_contribute = x;
        // 在对 N 进行二进制拆分的同时计算答案
        while (N > 0) {
            if (N % 2 == 1) {
                // 如果 N 二进制表示的最低位为 1，那么需要计入贡献
                ans *= x_contribute;
            }
            // 将贡献不断地平方
            x_contribute *= x_contribute;
            // 舍弃 N 二进制表示的最低位，这样我们每次只要判断最低位即可
            N /= 2;
        }
        return ans;
    }


    //49 x 的平方根
    public static int mySqrt(int x) {
        //简单方法,不能用整数
        /*double ans=1;
        while ((ans * ans) <= x) {
            ans++;
        }
       return (int)ans-1;*/
        //二分法
        double left = 1, right = x;
        double mid;
        while (!((left + 1) * (left + 1) > x)) {
            mid = Math.floor((left + right) / 2);
            double temp = mid * mid;
            if (temp > x) {
                right = (int) mid;
            } else if (temp < x) {
                left = (int) mid;
            } else return (int) mid;
        }
        return (int) left;
    }

    //48爬楼梯(求方法种数)每次你可以爬 1 或 2
    public static int climbStairs(int n) {
        //n=1,1, n=2,2, n=3,3, n=4,5, n=5 8等于前两个之和（可以用递归,但是栈会溢出或超出时间限制）
        //假如n-1层的时候有k种，n-2的时候有m种那么到k+m种了
        if (n == 2) return 2;
        if (n == 1) return 1;
        int sum = 0;
        // return climb(n, sum);
        int n_2 = 1, n_1 = 2;//用两个指针代替递归
        for (int i = 3; i <= n; i++) {
            sum = n_2 + n_1;
            n_2 = n_1;
            n_1 = sum;
        }
        return sum;


    }

    public static int climb(int n, int sum) {
        if (n == 3) return sum = 3;
        if (n == 2) return sum = 2;
        if (n == 1) return sum = 1;
        sum = climb(n - 1, sum) + climb(n - 2, sum);
        return sum;
    }

    //47字母异位
    public static boolean isAnagram(String s, String t) {
        //使用哈希表，先把一个存进去，键位char值为个数，再一个循环来递减(耗时比较长)
        /*int len=s.length(),lenT=t.length();
        if(len!=lenT)return false;
        HashMap<Character, Integer> map1 = new HashMap<>();
        for (int i = 0; i < len; i++) {
            char cs=s.charAt(i);
            if(!map1.containsKey(cs)){
                map1.put(cs,1);
            }else {
                int num=map1.get(cs);
                map1.put(cs,++num);
            }
        }
        for (int i = 0; i < lenT; i++) {
            char ct=t.charAt(i);
            //这一步可以简化（table.put(ch, table.getOrDefault(ch, 0) - 1);）
            if(!map1.containsKey(ct)){
                return false;
            }
            int er=map1.get(ct);
            if(er==0)return false;
            map1.put(ct,--er);
        }
        return true;*/
        //使用一个26长度的数组表示a-z，值为相应字母个数，一个循环存值，一个循环减值并判断
        //这种方法也称为哈希表，不是指hashmap那种
        int len = s.length(), lenT = t.length();
        if (len != lenT) return false;
        int[] temp = new int[26];
        for (int i = 0; i < len; i++) {
            temp[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < t.length(); i++) {
            if (--temp[t.charAt(i) - 'a'] < 0) return false;
        }
        return true;
        //也可以分别转为数组后排序用arrays.equals比较
    }

    //46单词规律
    public static boolean wordPattern(String pattern, String s) {
        int n = pattern.length();
        String[] splitP = s.split(" +");
        int m = splitP.length;
        if (m != n) {
            return false;
        }
        HashMap<String, Character> strP = new HashMap<>();
        HashMap<Character, String> cs = new HashMap<>();
        for (int i = 0; i < m; i++) {
            String pp = splitP[i];
            char ps = pattern.charAt(i);
            if ((strP.containsKey(pp) && strP.get(pp) != ps) || (cs.containsKey(ps) && !cs.get(ps).equals(pp)))
                return false;
            strP.put(pp, ps);
            cs.put(ps, pp);
        }
        return true;


    }

    //同构字符串
    public static boolean isIsomorphic(String s, String t) {
        //先判断两个字符长度是否相同
        //设计两个计数器，一次遍历
        int m = s.length(), n = t.length();
        if (m != n) return false;
        /*int c1=0,c2=0;
        for (int i = 1; i < m; i++) {
            c1= s.charAt(i)==s.charAt(i-1)?0:1;
            c2= t.charAt(i)==t.charAt(i-1)?0:1;
            //不能简单的判断前后，而是要考虑全局比如ABAB结构
            if(c1!=c2)return false;
        }
        return true;*/
        //考虑用两个哈希表来
        HashMap<Character, Character> c1 = new HashMap<>();
        HashMap<Character, Character> c2 = new HashMap<>();
        for (int i = 0; i < m; i++) {
            char c3 = s.charAt(i);
            char c4 = t.charAt(i);
            if ((c1.containsKey(c3) && c1.get(c3) != c4) || (c2.containsKey(c4) && c2.get(c4) != c3)) {
                return false;
            }
            c1.put(c3, c4);
            c2.put(c4, c3);
        }
        return true;
    }

    //哈希表
    public static boolean canConstruct(String ransomNote, String magazine) {

        int m = ransomNote.length(), n = magazine.length();
        if (m > n) return false;
        HashMap<Character, Integer> init = new HashMap<>();
        for (int i = 0; i < n; i++) {
            char c = magazine.charAt(i);
            if (init.containsKey(c)) {
                init.put(c, init.get(c) + 1);
            } else init.put(c, 1);
        }
        for (int i = 0; i < m; i++) {
            char c = ransomNote.charAt(i);
            if (!init.containsKey(c)) {
                return false;
            } else if (init.get(c) == 0) {
                return false;
            } else init.put(c, init.get(c) - 1);
        }
        return true;
    }

    //45匹配返回下标
    public static int strStr(String haystack, String needle) {
        int len1 = haystack.length();
        int len2 = needle.length();
        if (len2 > len1) return -1;
        for (int i = 0; i < len1; i++) {
            int index = 0;
            while (index < len2 && (i + index) < len1 && needle.charAt(index) == haystack.charAt(i + index)) {
                index++;
            }
            if (index == len2) return i;
        }
        return -1;
    }

    //43反转字符串中的单词
    public static String reverseWords(String s) {
        String trim = s.trim();
        Deque<String> d = new ArrayDeque<String>();
        StringBuilder word = new StringBuilder();
        int left = 0, right = trim.length() - 1;
        while (left <= right) {
            char c = trim.charAt(left);
            if ((word.length() != 0) && (c == ' ')) {
                // 将单词 push 到队列的头部
                d.offerFirst(word.toString());
                word.setLength(0);
            } else if (c != ' ') {
                word.append(c);
            }
            ++left;
        }
        d.offerFirst(word.toString());
        return String.join(" ", d);

        //使用string内置方法实现
        /*String[] str = s.split(" +");
        StringBuffer ans = new StringBuffer();
        for (int i = str.length-1; i >= 0; i--) {
            ans.append(str[i]+" ");
        }
        return ans.toString().trim();*/
    }

    //42最长公共前缀
    public static String longestCommonPrefix(String[] strs) {
        String str = strs[0];
        StringBuffer ans = new StringBuffer();
        int len = strs.length;
        if (len <= 1) {
            return str;
        }
        for (int i = 1; i < len; i++) {
            int index = 0;
            ans.delete(0, ans.length());
            while (index < str.length() && index < strs[i].length() && strs[i].charAt(index) == str.charAt(index)) {
                ans.append(str.charAt(index));
                index++;
            }
            str = ans.toString();
            if (str.length() == 0) return "";
        }
        return str;
    }

    //41最后一个单词的长度
    public static int lengthOfLastWord(String s) {
        String[] s1 = s.split(" ");
        return s1[s1.length - 1].length();
    }

    //40两数之和
    public static int[] twoSum(int[] numbers, int target) {
        int[] res = new int[2];
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            int temp = numbers[left] + numbers[right];
            if (temp > target) {
                while (left < right && numbers[right] == numbers[--right]) ;
            } else if (temp < target) {
                while (left < right && numbers[left] == numbers[++left]) ;
            } else {
                res[0] = left + 1;
                res[1] = right + 1;
                break;
            }
        }
        return res;
    }

    //39数转换罗马***
    public static String intToRoman(int num) {
        /* int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] symbols = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        StringBuffer roman = new StringBuffer();
        for (int i = 0; i < values.length; ++i) {
            int value = values[i];
            String symbol = symbols[i];
            while (num >= value) {
                num -= value;
                roman.append(symbol);
            }
            if (num == 0) {
                break;
            }
        }
        return roman.toString();*/
        //硬编码
        String[] thousands = {"", "M", "MM", "MMM"};
        String[] hundreds = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] ones = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};

        StringBuffer roman = new StringBuffer();
        roman.append(thousands[num / 1000]);
        roman.append(hundreds[num % 1000 / 100]);
        roman.append(tens[num % 100 / 10]);
        roman.append(ones[num % 10]);
        return roman.toString();

    }

    //38罗马数转换
    public static int romanToInt(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        int len = s.length();
        int sum = 0;
        for (int i = 0; i < len; i++) {
            int a = map.get(s.charAt(i));
            int b = 0;
            if (i < len - 1) {
                b = map.get(s.charAt(i + 1));
            }
            if (a >= b) sum = sum + a;
            else {
                sum = sum + b - a;
                i++;
            }
        }
        return sum;
    }

    //37除自身以外的乘积
    public static int[] productExceptSelf(int[] nums) {
        //两个数组,时空复杂度都是n
       /* int len=nums.length;
        int[] L = new int[len];
        int[] R = new int[len];
        int res[]=new int[len];
        // L[i] 为索引 i 左侧所有元素的乘积
        // 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
        L[0]=1;
        for (int i = 1; i < len; i++) {
            L[i]=nums[i-1]*L[i-1];
        }
        R[len-1]=1;
        for (int i = len-2; i >= 0; i--) {
            R[i]=nums[i+1]*R[i+1];
        }
        for (int i = 0; i < len; i++) {
            res[i]=L[i]*R[i];
        }
        return res;*/
        //空间复杂度为1
        int len = nums.length;
        int[] res = new int[len];
        // answer[i] 表示索引 i 左侧所有元素的乘积
        // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
        res[0] = 1;
        for (int i = 1; i < len; i++) {
            //现在充当左边的乘积
            res[i] = res[i - 1] * nums[i - 1];
        }
        int temp = 1;
        for (int i = len - 2; i >= 0; i--) {
            temp = temp * nums[i + 1];
            res[i] *= temp;
        }
        return res;

    }

    //36跳跃游戏
    public static int hIndex(int[] citations) {
        //排序
        Arrays.sort(citations);
        int h = 0;
        for (int i = citations.length - 1; i >= 0; i--) {
            if (citations[i] >= h) {
                h++;
            } else break;
        }
        return Math.min(h, citations[citations.length - h]);
        //计数排序
        //二分搜索

    }

    //35跳跃游戏
    public static boolean canJump(int[] nums) {
        //贪心
        int len = nums.length;
        int end = 0;
        int jump = 0;
        int step = 0;
        for (int i = 0; i < len - 1; i++) {
            jump = Math.max(jump, i + nums[i]);
            if (i == end) {
                end = jump;
                step++;
            }
        }
        return false;
        /*int count=1;
        if(nums.length<2)return true;
        for (int i = nums.length-2; i >0; i--) {
            if(nums[i]<count){
                count++;
            }else count=1;
        }
        if(nums[0]>=count)return true;
        return false;*/
    }

    //34买卖股票的最佳时机 II
    public static int maxProfit2(int[] prices) {
        int count = 0;
        int low = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > low) {
                count += prices[i] - low;
            }
            low = prices[i];
        }
        return count;
    }
    //33轮转数组

    public static void rotate(int[] nums, int k) {
        //翻转数组
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
        //简单的方法
        /*int len=nums.length;
        if(len==1)return;
        if(k>len){k=k%len;}
        int[] ints = Arrays.copyOfRange(nums, len-k, len);
        for (int i = len-1; i >=k; i--) {
            if(i>=k){
                nums[i]=nums[i-k];
            }
        }
        for (int i = 0; i < k; i++) {
            nums[i]=ints[i];
        }*/
        //上面方法简化
       /* int n = nums.length;
        int[] newArr = new int[n];
        for (int i = 0; i < n; ++i) {
            newArr[(i + k) % n] = nums[i];
        }
        System.arraycopy(newArr, 0, nums, 0, n);*/

    }

    public static void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }
    //32多数元素 II

    public static List<Integer> majorityElement3(int[] nums) {
        //摩尔投票法
        ArrayList<Integer> list = new ArrayList<>();
        int ele1 = 0;
        int ele2 = 0;
        int v1 = 0, v2 = 0;
        for (int num : nums) {
            if (v1 > 0 && num == ele1) {//如果该元素为第一个元素，则计数加1
                v1++;
            } else if (v2 > 0 && num == ele2) {//如果该元素为第二个元素，则计数加1
                v2++;
            } else if (v1 == 0) {// 选择第一个元素
                ele1 = num;
                v1++;
            } else if (v2 == 0) {// 选择第二个元素
                ele2 = num;
                v2++;
            } else {//如果三个元素均不相同，则相互抵消1次
                v1--;
                v2--;
            }
        }
        int c1 = 0, c2 = 0;
        for (int num : nums) {
            if (num == ele1) {
                c1++;
            } else if (num == ele2) {
                c2++;
            }
        }
        if (v1 > 0 && c1 > nums.length / 3) {
            list.add(ele1);
        }
        if (v2 > 0 && c2 > nums.length / 3) {
            list.add(ele2);
        }
        return list;
   /*  该方法比较占空间   Arrays.sort(nums);
        ArrayList<Integer> list = new ArrayList<>();
        int count = 0;
        int len = nums.length;
        int temp=nums[0];
        for (int i = 0; i < len; i++) {
            if(nums[i]==temp){
                count++;
            }else {
                count=1;
                temp=nums[i];
            }
            if(count>len/3&&!list.contains(nums[i])){
                list.add(nums[i]);
            }
        }
        return list;*/
    }
    //31删除有序数组中的重复项 II

    public static int removeDuplicates(int[] nums) {
        int len = nums.length;
        int low = 1, fast = 1;
        int count = 0;
        if (len == 0) return 0;
        for (int i = 0; i < len - 1; i++) {
            if (nums[fast] != nums[fast - 1]) {
                nums[low] = nums[fast];
                ++low;
                count = 0;
            } else if (count < 1) {
                count++;
                nums[low] = nums[fast];
                ++low;
            }
            ++fast;
        }
        return low;

    }
    //30提莫攻击

    public static int findPoisonedDuration(int[] timeSeries, int duration) {
        int count = 0;
        int len = timeSeries.length;
        for (int i = 0; i < len - 1; i++) {
            int e = timeSeries[i + 1] - timeSeries[i];
            if (e <= duration) {
                count += e;
            } else count += duration;
        }
        return count + duration;
    }
    //29最大连续1

    public static int findMaxConsecutiveOnes(int[] nums) {
        int len = nums.length;
        int max = 0;
        int count = 0;
        for (int i = 0; i < len; i++) {
            if (nums[i] == 1) {
                count++;
            } else {
                max = Math.max(max, count);
                count = 0;
            }
        }
        return Math.max(count, max);

    }
    //28边界着色

    public static int[][] colorBorder(int[][] grid, int row, int col, int color) {
        //广度优先
        int[] dx = {0, 1, 0, -1};
        int[] dy = {1, 0, -1, 0};

        //初始颜色
        int initColor = grid[row][col];
        int m = grid.length, n = grid[0].length;
        int[][] temp = new int[m][n];
        temp[row][col] = 1;
        Queue<int[]> queue = new LinkedList<>();
        //储存相连色块位置的队列
        queue.offer(new int[]{row, col});
        while (!queue.isEmpty()) {
            int[] place = queue.poll();
            int tx = place[0], ty = place[1];
            for (int i = 0; i < 4; i++) {
                int x = tx + dx[i], y = ty + dy[i];
                if (x >= 0 && x < m && y >= 0 && y < n && temp[x][y] >= 1) {
                    temp[x][y]++;
                    continue;
                }
                if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == initColor) {
                    queue.offer(new int[]{x, y});
                    temp[x][y]++;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (temp[i][j] > 0 && temp[i][j] < 4) {
                    grid[i][j] = color;
                }
            }
        }
        if (temp[row][col] == 4) grid[row][col] = color;
        return grid;

    }
    //27图像渲染

    public static int[][] floodFill(int[][] image, int sr, int sc, int color) {
        //递归会内存爆炸
        /* int []dx={0,1,0,-1};
        int[]dy={1,0,-1,0};//上下左右相连的坐标增量；
        int m=image.length,n=image[0].length;
        int num=image[sr][sc];//初始色
        if(image[sr][sc]==num){
            image[sr][sc]=color;
            for (int k = 0; k < 4; k++) {
                int tx=sr+dx[k];
                int ty=sc+dy[k];
                if(tx>=0&&tx<m&&ty>=0&&ty<n&&image[tx][ty]==num){
                    floodFill(image,tx,ty,color);
                }
            }
        }
        return image;*/
        //广度优先搜索
        int[] dx = {0, 1, 0, -1};
        int[] dy = {1, 0, -1, 0};
        //初始颜色
        int intColor = image[sr][sc];
        if (intColor == color) return image;
        //数组大小
        int n = image.length, m = image[0].length;
        Queue<int[]> queue = new LinkedList<>();
        //储存相连色块位置的队列
        queue.offer(new int[]{sr, sc});
        //防止重复
        image[sr][sc] = color;
        while (!queue.isEmpty()) {
            int[] place = queue.poll();
            int tx = place[0], ty = place[1];
            for (int i = 0; i < 4; i++) {
                int x = tx + dx[i], y = ty + dy[i];
                if (x >= 0 && x < n && y >= 0 && y < m && image[x][y] == intColor) {
                    queue.offer(new int[]{x, y});
                    image[x][y] = color;
                }
            }
        }
        return image;


    }
    //26岛屿周长

    public static int islandPerimeter(int[][] grid) {
        int len = grid.length, len2 = grid[0].length;
        int ans = 0;
        int[] dx = {0, 1, 0, -1};
        int[] dy = {1, 0, -1, 0};
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len2; j++) {
                if (grid[i][j] == 1) {
                    int cn = 0;
                    for (int k = 0; k < 4; k++) {
                        int tx = i + dx[k];
                        int ty = j + dy[k];
                        if (tx < 0 || tx >= len || ty < 0 || ty >= len2 || grid[tx][ty] == 0) {
                            cn++;
                        }
                    }
                    ans += cn;
                }
            }

        }
        return ans;

    }
    //25分发饼干

    public static int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int m = g.length, n = s.length;
        int count = 0;
        for (int i = 0, j = 0; i < m && j < n; i++, j++) {
            while (j < n && g[i] > s[j]) {
                j++;
            }
            if (j < n) count++;
        }
        return count;

    }
    //24找到所有数组中消失的数字

    public static List<Integer> findDisappearedNumbers(int[] nums) {
        int len = nums.length;
        int[] ints = new int[len + 1];
        for (int num : nums) {
            ints[num] = 1;
        }
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 1; i < len + 1; i++) {
            if (ints[i] == 0) list.add(i);
        }
        return list;
    }
    //23第三大的数

    public static int thirdMax(int[] nums) {
        /*
        时间复杂度大
        int[] ints = Arrays.stream(nums).distinct().toArray();
        Arrays.sort(ints);
        if(ints.length>=3) return ints[ints.length-3];
        else return ints[ints.length-1];
        */
        long max = Long.MIN_VALUE, mid = Long.MIN_VALUE, min = Long.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > max) {
                min = mid;
                mid = max;
                max = nums[i];
            } else if (nums[i] < max && nums[i] > mid) {
                min = mid;
                mid = nums[i];
            } else if (nums[i] < mid && nums[i] > min) {
                min = nums[i];
            }
        }
        return min == Long.MIN_VALUE ? (int) max : (int) min;


    }
    //22四数之和

    public static List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        int len = nums.length;
        ArrayList<List<Integer>> lists = new ArrayList<>();
        if (nums == null || len < 4) return lists;
        for (int i = 0; i < len - 3; i++) {
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            if ((long) nums[i] + nums[len - 3] + nums[len - 2] + nums[len - 1] < target) continue;
            List<List<Integer>> lists1 = threeSum(Arrays.copyOfRange(nums, i + 1, len), target - nums[i]);
            if (!lists1.isEmpty()) {
                for (List<Integer> integers : lists1) {
                    integers.add(0, nums[i]);
                }
                lists.addAll(lists1);
            }
        }
        return lists;
    }
    //21三数之和最接近

    public static int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);//-2,-2,-1,0,1,1,2,3,4,5
        int best = 1000000;

        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int mid = i + 1, right = nums.length - 1;
            while (mid < right) {
                int sum = nums[i] + nums[mid] + nums[right];
                if (sum == target) {
                    return target;
                } else if (Math.abs(sum - target) < Math.abs(best - target)) {
                    best = sum;
                }
                if (sum < target) {
                    while (mid < right && nums[mid] == nums[++mid]) ;
                } else if (sum > target) {
                    while (mid < right && nums[right] == nums[--right]) ;
                }

            }
        }
        return best;
    }

    //20两数之和
    //20三数之和
    public static List<List<Integer>> threeSum(int[] nums, int targat) {
        Arrays.sort(nums);//-2,-2,-1,0,1,1,2,3,4,5
        ArrayList<List<Integer>> lists = new ArrayList<>();

        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int mid = i + 1, right = nums.length - 1;
            while (mid < right) {
                int sum = nums[i] + nums[mid] + nums[right];
                if (sum < targat) {
                    while (mid < right && nums[mid] == nums[++mid]) ;
                } else if (sum > targat) {
                    while (mid < right && nums[right] == nums[--right]) ;
                } else {
                    lists.add(new ArrayList<Integer>(Arrays.asList(nums[i], nums[mid], nums[right])));
                    while (mid < right && nums[mid] == nums[++mid]) ;
                    while (mid < right && nums[right] == nums[--right]) ;
                }
            }
        }
        return lists;

    }

    //19盛最多水的容器
    public static int maxArea(int[] height) {
        int max = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            int min = Math.min(height[right], height[left]);
            int step = min * (right - left);
            max = Math.max(max, step);
            if (height[left] == min) {
                left++;
            } else right--;
        }
        return max;

    }

    //18数组交集1,2
    public static int[] intersection(int[] nums1, int[] nums2) {
        /*Set<Integer> set = Arrays.stream(nums1).boxed().collect(Collectors.toSet());
        int[] ints = Arrays.stream(nums2).distinct().filter(set::contains).toArray();
        return ints;*/
        /*HashMap<Integer, Integer> map1 = new HashMap<>();
        HashMap<Integer, Integer> map2 = new HashMap<>();
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums1.length; i++) {
            Integer put = map1.put(nums1[i], 1);
            if(put!=null){
                map1.put(nums1[i],put+1);
            }
        }
        for (int i = 0; i < nums2.length; i++) {
            Integer put = map2.put(nums2[i], 1);
            if(put!=null){
                map2.put(nums2[i],put+1);
            }
        }

        for(Map.Entry<Integer,Integer> entry: map1.entrySet()){
            Integer key = entry.getKey();
            Integer value = entry.getValue();
            Integer value2 = map2.get(key);
            if(value2!=null){
                int a=(value2>value?value:value2);
                for (int i = 0; i < a; i++) {
                    list.add(key);
                }
            }
        }
        int size = list.size();
        int[] res = new int[size];
        for (int i = 0; i < size; i++) {
            res[i]=list.get(i);
        }
        return res;*/
       /* 需要排序，Arrays.sort(nums1);
        Arrays.sort(nums2);
        int len1=nums1.length,len2=nums2.length;
        int len=Math.min(len1,len2);
        int []res=new int[len];
        int index1=0,index2=0,index=0;
        while (index1<len1&&index2<len2){
            if(nums1[index1]<nums2[index2]) {
                    index1++;
            }else if(nums1[index1]>nums2[index2]){
                index2++;
            }else {
                res[index]=nums1[index1];
                index2++;
                index1++;
                index++;
            }
             return Arrays.copyOfRange(res, 0, index);
        }*/
        //空间换时间
        int[] num1 = new int[1001];
        int[] res = new int[1001];
        for (int i : nums1) {
            num1[i]++;
        }
        int index = 0;
        for (int i : nums2) {
            if (--num1[i] >= 0) {
                res[index++] = i;
            }
        }
        return Arrays.copyOfRange(res, 0, index);
    }

    //17移动零
    public static void moveZeroes(int[] nums) {
        int len = nums.length;
        int left = 0;
        int sum = 0;
        for (int i = 0; i < len; i++) {
            if (nums[i] != 0) {
                nums[left] = nums[i];
                left++;
            } else sum++;
        }
        for (int i = 0; i < sum; i++) {
            nums[len - i - 1] = 0;
        }
    }

    //16汇总区间
    public static List<String> summaryRanges(int[] nums) {
       /* ArrayList<String> list = new ArrayList<>();
        int left = 0, len = nums.length;
        for (int i = 0; i < len - 1; i++) {
            if (nums[i] + 1 != nums[i + 1]) {
                if ((i - left) > 0) {
                    list.add('"' +""+ nums[left] + "->" + nums[i]+'"');
                    left = i + 1;
                }else {
                    list.add('"' +""+ nums[left] +'"');
                    left = i + 1;
                }
            }
        }
        if(left+1==len)list.add('"' +""+ nums[left] +'"');
        else  list.add('"' +""+ nums[left] + "->" + nums[len-1]+'"');
        return list;*/
        ArrayList<String> list = new ArrayList<>();
        int left = 0, len = nums.length;
        for (int i = 0; i < len - 1; i++) {
            if (nums[i] + 1 != nums[i + 1]) {
                StringBuffer temp = new StringBuffer(Integer.toString(nums[left]));
                if ((i - left) > 0) {
                    temp.append("->");
                    temp.append(Integer.toString(nums[i]));
                }
                list.add(temp.toString());
                left = i + 1;
            }
        }
        StringBuffer temp2 = new StringBuffer(Integer.toString(nums[left]));
        if (left + 1 != len) {
            temp2.append("->");
            temp2.append(nums[len - 1]);
        }
        list.add(temp2.toString());
        return list;
    }

    //15丢失的数
    public static int missingNumber(int[] nums) {
        int len = nums.length;
        int sum1 = 0, sum2 = len;
        for (int i = 0; i < len; i++) {
            sum1 += nums[i];
            sum2 += i;
        }
        return sum2 - sum1;
    }

    //14存在重复元素2
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        /*        HashSet<Integer> set = new HashSet<>();
        //滑动窗口
        for (int i = 0; i < nums.length; i++) {
            if(i>k){
                set.remove(nums[i-k-1]);
            }
            if(!set.add(nums[i])){
                return true;
            }
        }
        return false;*/

        //hashmap
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (map.containsKey(num) && i - map.get(num) <= k) {
                return true;
            }
            map.put(nums[i], i);
        }
        return false;
    }

    public static int extracted(int[] nums) {
       /* ArrayList<Integer> rest = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            boolean flag=true;
            for (int j = 0; j < i; j++) {
                if (nums[i]== nums[j]){
                    flag=false;
                    break;
                }
            }
            if(flag){
                rest.add(nums[i]);
            }
        }
        Object[] re = rest.toArray();
        System.out.println(rest.toString());

        return  rest.size();*/
        int n = nums.length;
        if (n == 0)
            return 0;
        int fast = 1;
        int slow = 1;
        for (int i = 0; i < n - 1; i++) {
            if (nums[fast] != nums[fast - 1]) {
                nums[slow] = nums[fast];
                ++slow;
            }
            ++fast;
        }
        return slow;
    }

    public static int removeElement(int[] nums, int val) {
        int len = nums.length;

        if (len == 0) return 0;
        int left = 0;

        for (int i = 0; i < len; i++) {
            if (nums[i] != val) {
                nums[left] = nums[i];
                left++;
            }
        }
        return left;
    }

    //11数组插入
    public static int searchInsert(int[] nums, int target) {
        int len = nums.length;
        /*采用二分查找会快一些*/
        /*int res = 0;
        if(nums[len-1] < target)return len;
        for (int i = 0; i < len; i++) {
            if (nums[i] >= target ) {
                res = i;
                break;
            }
        }
        return res;*/
        int left = 0, right = len - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] < target) {
                left = mid + 1;
            } else right = mid - 1;

        }
        return left;

    }

    //10数组加一
    public static int[] plusOne(int[] digits) {
        int len = digits.length;

        for (int i = len - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i] = digits[i] + 1;
                return digits;
            } else {
                digits[i] = 0;
            }
        }
        int[] res = new int[len + 1];
        res[0] = 1;
        return res;
    }

    //9合并数组
    public static void merge(Integer[] nums1, int m, int[] nums2, int n) {
      /*
        直接使用数组快速排序
        int left = 0;
        if (n == 0) return;
        for (int i = 0; i < n; i++) {
            nums1[m + i] = nums2[i];
        }
        Arrays.sort(nums1);*/
        /*双指针*/
        int p1 = n - 1, p2 = m - 1;
        int tail = m + n - 1;
        int cur;
        while (p2 >= 0 || p1 >= 0) {
            if (p1 == -1) {
                cur = nums2[p2--];
            } else if (p2 == -1) {
                cur = nums1[p1--];
            } else if (nums1[p1] < nums2[p2]) {
                cur = nums1[p1--];
            } else cur = nums1[p2--];
            nums1[tail--] = cur;
        }


    }

    /*8数组快速排序方法
     * 先确定基准数可以随机也可以指定，每次大循环得到一个数使得这个数前面的都小于它后面的都大于它
     * 再使用递归最后进行排序*/
    public static void quick_sort(int[] s, int l, int r) {
        //l,r为前后基准数位置
        if (l < r) {
            int i = l, j = r, x = s[l];
            while (i < j) {
                while (i < j && s[j] >= x) {// 从右向左找第一个小于x的数
                    j--;
                    if (i < j) {
                        s[i++] = s[j];
                    }
                }
                while (i < j && s[i] > x) {
                    i++;
                    if (i < j)
                        s[j--] = s[i];// 从左向右找第一个大于x的数
                }
            }
            s[i] = x;
            quick_sort(s, l, i - 1);
            quick_sort(s, i + 1, r);
        }
    }

    //7数组转化为平衡二叉树
    public static TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);
    }

    //6
    private static TreeNode helper(int[] nums, int left, int right) {
        if (left > right) return null;
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = helper(nums, left, mid - 1);
        root.right = helper(nums, mid + 1, right);
        return root;
        /*
        * public void postOrderRecur(Node root) {
		if (root == null) {
			return;
		}
		* System.out.print(root.data + " -> ");前序遍历
		postOrderRecur(root.left);
		* System.out.print(root.data + " -> ");中序遍历
		postOrderRecur(root.right);
		System.out.print(root.data + " -> ");后序遍历
	}
*/
    }

    //5杨辉三角
    public static List<List<Integer>> generate(int numRows) {
        ArrayList<List<Integer>> lists = new ArrayList<>();
        for (int i = 0; i < numRows + 1; i++) {
            ArrayList<Integer> list_temp = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    list_temp.add(1);
                } else list_temp.add(lists.get(i - 1).get(j - 1) + lists.get(i - 1).get(j));
            }
            lists.add(list_temp);
        }
        return lists;
    }

    //4买卖股票最佳时机
    public static int maxProfit(int[] prices) {
        //时间复杂度太大是阶乘
        /*int max=0,len=prices.length;
        for (int i = 0; i < len; i++) {
            for (int j = i; j < len; j++) {
                int e = prices[j] - prices[i];
                if(e>max){
                    max=e;
                }
            }
        }
        return max;*/
        int len = prices.length;
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int i = 0; i < len; i++) {
            if (prices[i] < min) {
                min = prices[i];
            } else if (prices[i] - min > max) {
                max = prices[i] - min;
            }
        }
        return max;

    }

    //3只出现一次的数字
    //用异或来，
    public static int singleNumber(int[] nums) {
        int res = nums[0];//存储结果，遍历一次每个都与其进行异或运算，由于只有一个单的，其他都是双的，遍历异或后就是那个单的
        if (nums.length > 1) {
            for (int i = 1; i < nums.length; i++) {
                res = res ^ nums[i];
            }
        }
        return res;
    }

    //2多数元素>n/2
    public static int majorityElement(int[] nums) {
      /*  Arrays.sort(nums);
      return nums[nums.length/2];*/
        //使用哈希表来
        HashMap<Integer, Integer> map = new HashMap<>();
        int len = nums.length;
        map.put(nums[0], 1);
        for (int i = 1; i < len; i++) {
            if (map.containsKey(nums[i])) {
                map.put(nums[i], map.get(nums[i]) + 1);
            } else map.put(nums[i], 1);
        }
        Map.Entry<Integer, Integer> major = null;
        for (Map.Entry<Integer, Integer> entry1 : map.entrySet()) {
            if (major == null || entry1.getValue() > major.getValue())
                major = entry1;
        }
        return major.getKey();
    }

    //1存在重复元素
    public static boolean containsDuplicate(int[] nums) {
        //1hashset
        HashSet<Integer> setNums = new HashSet<>();
        for (int num : nums) {
            if (!setNums.add(num)) {
                return true;
            }
        }
        return false;

    }

}
