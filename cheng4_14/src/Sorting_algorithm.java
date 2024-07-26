import java.util.Arrays;

public class Sorting_algorithm {
    public static void main(String[] args) {

    }
    /* 冒泡排序 */
    public static void BubbleSort(int arr[], int length) {
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length -  i - 1; j++)
            {
                //把剩余数的最大值移到最后
                if (arr[j] > arr[j + 1])
                {
                    int temp;
                    temp = arr[j + 1];
                    arr[j + 1] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }
    /* 选择排序 */
    public static void SelectionSort(int arr[], int length) {
        int index, temp;
        for (int i = 0; i < length; i++)
        {
            index = i;
            for (int j = i + 1; j < length; j++)
            {
                //把剩余数的最小值移到前面
                if (arr[j] < arr[index])
                    index = j;
            }
            if (index != i)
            {
                temp = arr[i];
                arr[i] = arr[index];
                arr[index] = temp;
            }
        }
    }
    /* 插入排序 */
    public static void InsertSort(int arr[], int length) {
        for (int i = 1; i < length; i++)
        {
            int j;
            if (arr[i] < arr[i - 1])
            {
                int temp = arr[i];
                for (j = i - 1; j >= 0 && temp < arr[j]; j--)
                {
                    arr[j + 1] = arr[j];
                }
                arr[j + 1] = temp;
            }
        }
    }
    // 快速排序
    public static void QuickSort(int arr[], int start, int end) {
        if (start >= end)
            return;
        int i = start;
        int j = end;
        // 基准数
        int baseval = arr[start];
        while (i < j)
        {
            // 从右向左找比基准数小的数
            while (i < j && arr[j] >= baseval) {
                j--;
            }
            if (i < j) {
                arr[i] = arr[j];
                i++;
            }
            // 从左向右找比基准数大的数
            while (i < j && arr[i] < baseval) {
                i++;
            }
            if (i < j) {
                arr[j] = arr[i];
                j--;
            }
        }
        // 把基准数放到i的位置
        arr[i] = baseval;
        // 递归
        QuickSort(arr, start, i - 1);
        QuickSort(arr, i + 1, end);
    }
    //希尔排序是插入排序的改进
    //桶排序
    //基数排序（桶排序衍生）
    /*基数排序：根据键值的每位数字来分配桶；
    计数排序：每个桶只存储单一键值；
    桶排序：每个桶存储一定范围的数值；*/

}
