package verifier;

import java.util.List;
import java.util.Set;

import graph.PrecedenceGraph;
import graph.TxnNode;
import util.Pair;
import util.ChengLogger;

public class Constraint {
	public Set<Pair<Long, Long>> edge_set1, edge_set2;
	public List<TxnNode> chain_1, chain_2;

	public Constraint(Set<Pair<Long, Long>> edges1, Set<Pair<Long, Long>> edges2, List<TxnNode> chain1,
			List<TxnNode> chain2) {
		edge_set1 = edges1;
		edge_set2 = edges2;
		chain_1 = chain1;
		chain_2 = chain2;
	}

	public String toString(PrecedenceGraph g) {
		return toString(g, false);
	}

	public String toString(PrecedenceGraph g, boolean detail) {
		StringBuilder sb = new StringBuilder();
		sb.append("Chain 1:\n");
		for (TxnNode tx : chain_1) {
			if (detail) {
				sb.append("  " + tx.toString2() + "\n");
			} else {
				sb.append("  " + tx.toString3() + "\n");
			}
		}
		sb.append("Chain 2:\n");
		for (TxnNode tx : chain_2) {
			if (detail) {
				sb.append("  " + tx.toString2() + "\n");
			} else {
				sb.append("  " + tx.toString3() + "\n");
			}
		}
		sb.append("Edges 1:\n");
		for (Pair<Long, Long> e : edge_set1) {
			TxnNode src = g.getNode(e.getFirst());
			TxnNode dst = g.getNode(e.getSecond());
			sb.append("  " + src.toString3() + "->" + dst.toString3() + "\n");
		}
		sb.append("Edges 2:\n");
		for (Pair<Long, Long> e : edge_set2) {
			TxnNode src = g.getNode(e.getFirst());
			TxnNode dst = g.getNode(e.getSecond());
			sb.append("  " + src.toString3() + "->" + dst.toString3() + "\n");
		}
		return sb.toString();
	}

	@Override
	public String toString() {
		ChengLogger.println("using toString made by me");
    		StringBuilder sb = new StringBuilder();
   		sb.append("edge_set1: ").append(edge_set1.toString()).append("\n");
    		sb.append("edge_set2: ").append(edge_set2.toString()).append("\n");
    		sb.append("chain_1: ").append(chain_1.toString()).append("\n");
    		sb.append("chain_2: ").append(chain_2.toString()).append("\n");
    		return sb.toString();
	}

	//added part
	public long[][] getTxidOfEdges(PrecedenceGraph g){
		long[][] edgesArray = new long[2][2];
		int countEdges1 = 0;
		for (Pair<Long, Long> e : edge_set1) {
			TxnNode src = g.getNode(e.getFirst());
			TxnNode dst = g.getNode(e.getSecond());
			edgesArray[0][0] = src.getTxnid();
			edgesArray[0][1] = dst.getTxnid();
			countEdges1 += 1;
		}
		if (countEdges1!=1) System.out.println("Edges error");
		int countEdges2 = 0;
		for (Pair<Long, Long> e : edge_set2) {
			TxnNode src = g.getNode(e.getFirst());
			TxnNode dst = g.getNode(e.getSecond());
			edgesArray[1][0] = src.getTxnid();
			edgesArray[1][1] = dst.getTxnid();
			countEdges2 += 1;
		}
		if (countEdges2!=1) System.out.println("Edges error");
		return edgesArray;
	}
}
