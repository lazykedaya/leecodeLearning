import java.util.*;

public class RandomizedSet {
    //变长数组nums可以O（1）时间内完成随机获取元素，但是无法再O（1）时间内判断元素是否存在
    //hash表可以在O（1）时间内完成插入和删除操作，但是无法根据下标定位到特定元素，需要两者结合
    List<Integer> nums;
    Map<Integer,Integer>indices;
    Random random;
    public RandomizedSet() {
        nums=new ArrayList<>();
        indices=new HashMap<>();
        random=new Random();
    }

    public boolean insert(int val) {
        //如果存在就返回插入失败
        if(indices.containsKey(val)){
            return false;
        }
        //如果不存在，存入nums，并把nums中的index存到indices中
        int index=nums.size();
        nums.add(val);
        indices.put(val,index);
        return true;
    }

    public boolean remove(int val) {
        //不存在就返回false
        if(!indices.containsKey(val)){
            return false;
        }
        //不存在得把nums和indices里面的有关删除，同时要保证其他nums值的位置与indices里面对应键值的对应关系
        int index=indices.get(val);
        //将nums最后一个数移到删除的位置；
        int last=nums.get(nums.size()-1);
        nums.set(index,last);
        //更新nums之前最后一个数index在indices里面的键值
        indices.put(last,index);
        //最后删除nums最后一个数，以及indices里的val值
        nums.remove(nums.size()-1);
        indices.remove(val);
        return true;
    }

    public int getRandom() {
        return nums.get(random.nextInt(nums.size()));
    }
    }
