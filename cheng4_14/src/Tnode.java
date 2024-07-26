public class Tnode {
    public int val;
    public Tnode left;
    public Tnode right;
    public Tnode next;

    public Tnode() {}

    public Tnode(int _val) {
        val = _val;
    }

    public Tnode(int _val, Tnode _left, Tnode _right, Tnode _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}
