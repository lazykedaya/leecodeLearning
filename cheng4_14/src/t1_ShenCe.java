import java.util.Random;

public class t1_ShenCe {
    public static void main(String[] args) {
        int arr[] = new int[30];
        Random random = new Random();

        // 循环遍历数组并随机填充数据
        for (int i = 0; i < arr.length; i++) {
            arr[i] = random.nextInt(100); // 生成0-99之间的随机数
            System.out.print(arr[i] + ", ");
        }
        //x是指标，我只需要最大的不需要保存其他数据
       /* int x=0;
       //这个复杂度是n2了
        for (int i = 1; i < arr.length; i++) {
            for (int j = 0; j < i; j++) {
                int temp;
                temp=x;
                x=arr[i]+arr[j]-(i-j);
                x= Math.max(x, temp);
            }
        }
        System.out.println();
        System.out.println(x);*/

        //下面的降低了复杂度
        int preBest = arr[0];
        int ans = 0;
        for (int i = 0; i < arr.length; i++) {
            ans = Math.max(ans, arr[i] - i + preBest);
            preBest = Math.max(preBest, arr[i] + i);
        }
        System.out.println(ans);

    }
}
//12, 57, 26, 28, 91, 16, 64, 46, 60, 13, 97, 99, 30,
//80, 68, 47, 43, 94, 61, 76, 34, 14, 52, 66, 41, 53, 67, 96, 18, 31,