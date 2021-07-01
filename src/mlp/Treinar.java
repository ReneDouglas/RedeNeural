package mlp;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class Treinar {
	
	private final double TAXA_APREND = 0.6;
	private final double MOMENTO = 0.2;
	private ArrayList<Double> entradas_treinamento;
	private ArrayList<Double> entradas_validacao;
	private ArrayList<Double> entradas_teste;
	private ArrayList<Double> saidas_treinamento;
	private ArrayList<Double> saidas_validacao;
	private ArrayList<Double> saidas_teste;
	private int num_padroes;
	private double erro_treinamento = 0.0;
	private double erro_validacao = 0.0;
	private double erro_teste = 0.0;
	private ArrayList<Neuronio> neuronios;
	private MLP mlp;
	private BackPropagation bp;
	private Camada oculta;
	private Camada saida;
	private double entradas_trein[][], entradas_val[][], entradas_test[][];
	//private double saidas_trein[][], saidas_val[][], saidas_test[][];
	private FileWriter arq;
	private PrintWriter gravar;
	private SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy hh:mm"); 
	
	public Treinar() {
		
		sdf = new SimpleDateFormat("dd/MM/yyyy hh:mm");
		System.out.println("Início: "+sdf.format(new Date()));
		
		cabecalho();
		preparar();
		treinamento();
		//validacao();
		teste();
		
		informacoes();

	}
	
	private void treinamento(){
		
		//this.num_padroes = this.entradas_treinamento.size();
		this.num_padroes = this.entradas_trein.length;
		this.mlp.randomizarPesos(num_padroes(entradas_trein)/*this.num_padroes*/, this.entradas_trein[0].length/*this.entradas_treinamento.size()*/);
		
		int epoca = 0;
		int controle_trein = 0;
		boolean loop = true;
		double erro_validacao_anterior = 1.0;
		/*double saida_desejada_trein[] = new double[this.entradas_treinamento.size()];
		double saida_real_trein[] = new double[this.entradas_treinamento.size()];
		double saida_desejada_validacao[] = new double[this.entradas_validacao.size()];
		double saida_real_validacao[] = new double[this.entradas_validacao.size()];*/
		double saida_desejada_trein[] = new double[this.entradas_trein.length];
		double saida_real_trein[] = new double[this.entradas_trein.length];
		double saida_desejada_validacao[] = new double[this.entradas_val.length];
		double saida_real_validacao[] = new double[this.entradas_val.length];
		
		do{
			//this.num_padroes = this.entradas_treinamento.size();
			this.num_padroes = this.entradas_trein.length;
			while(controle_trein < this.num_padroes){
					
				for (int i = 0; i < oculta.neuronios.size(); i++) {
					if(oculta.neuronios.get(i).entradas.size() == 0){
						for (int j = 0; j < this.entradas_trein[controle_trein].length; j++) {
							oculta.neuronios.get(i).entradas.add(entradas_trein[controle_trein][j]);
						}	
						
					}
					else{
						for (int j = 0; j < this.entradas_trein[controle_trein].length; j++) {
							oculta.neuronios.get(i).entradas.set(j, entradas_trein[controle_trein][j]);
						}
						
					}
				}
				
				/*for (int i = 0; i < oculta.neuronios.size(); i++) {
					if(oculta.neuronios.get(i).entradas.size() == 0){
						oculta.neuronios.get(i).entradas.add(this.entradas_treinamento.get(controle_trein));
					}
					else{
						oculta.neuronios.get(i).entradas.set(0, this.entradas_treinamento.get(controle_trein));
					}
				}*/
				
				this.bp.setSaidaDesejada(this.saidas_treinamento.get(controle_trein));
				this.bp.propagacao();
				this.bp.retropropagacao();
				
				saida_desejada_trein[controle_trein] = this.bp.saidaDesejada;
				saida_real_trein[controle_trein] = this.mlp.saidaReal;
				
				for (int i = 0; i < this.mlp.camadas.size(); i++) {
					this.mlp.camadas.get(i).gradiente_local.clear();
				}
				controle_trein++;
			}
			
			if(controle_trein >= this.saidas_treinamento.size())controle_trein = 0;
			
			this.erro_treinamento = this.bp.erro_medio_quad(num_padroes(entradas_trein));//matriz completa
			this.bp.somatorio_energia.clear();
			
			// ********** VALIDACAO **********
			
			//this.num_padroes = this.entradas_validacao.size();
			this.num_padroes = this.entradas_val.length;
			int controle_val = 0;
			
			while(controle_val < this.num_padroes){
				
				/*for (int i = 0; i < oculta.neuronios.size(); i++) {
					if(oculta.neuronios.get(i).entradas.size() == 0){
						oculta.neuronios.get(i).entradas.add(this.entradas_validacao.get(controle_val));
					}
					else{
						oculta.neuronios.get(i).entradas.set(0, this.entradas_validacao.get(controle_val));
					}
				}*/
				
				for (int i = 0; i < oculta.neuronios.size(); i++) {
					if(oculta.neuronios.get(i).entradas.size() == 0){
						for (int j = 0; j < this.entradas_val[controle_val].length; j++) {
							oculta.neuronios.get(i).entradas.add(entradas_val[controle_val][j]);
						}	
						
					}
					else{
						for (int j = 0; j < this.entradas_val[controle_val].length; j++) {
							oculta.neuronios.get(i).entradas.set(j, entradas_val[controle_val][j]);
						}
						
					}
				}
				
				//this.bp.setSaidaDesejada(this.saidas_validacao.get(controle_val));
				this.bp.setSaidaDesejada(this.saidas_validacao.get(controle_val));
				this.bp.propagacao();
				
				//saida_desejada_validacao[controle_val] = this.bp.saidaDesejada;
				saida_desejada_validacao[controle_val] = this.bp.saidaDesejada;
				saida_real_validacao[controle_val] = this.mlp.saidaReal;
				
				for (int i = 0; i < this.mlp.camadas.size(); i++) {
					this.mlp.camadas.get(i).gradiente_local.clear();
				}
					
				controle_val++;
				
			}
			controle_val = 0;
			epoca ++;
			
			this.erro_validacao = this.bp.erro_medio_quad(num_padroes(entradas_val));
			this.bp.somatorio_energia.clear();
			
			if(epoca >= 100){
				if(epoca%10==0){//10000
					//System.out.println("ET: "+this.erro_treinamento+"\tErro Validação: "+this.erro_validacao+"\t epocas: "+epoca);
					this.gravar.printf("%d; %f %n", epoca, erro_validacao);
					if(this.erro_validacao >= erro_validacao_anterior) loop = false; 
					else erro_validacao_anterior = this.erro_validacao;
				}
			}else{
				if(epoca%1000==0){
					this.gravar.printf("%d; %f %n", epoca, erro_validacao);
				}
			}
			//this.gravar.printf("%d; %f %n", epoca, erro_validacao);
		
		}while(loop);
		
		try { arq.close(); } catch (IOException e) {e.printStackTrace();}
		
		System.out.println("----------- Treinamento -----------");
		for (int i = 0; i < saida_desejada_trein.length; i++) {
			System.out.println("Desejada: "+saida_desejada_trein[i]+" Real: "+saida_real_trein[i]);
		}
		System.out.println("Epocas: "+epoca+" Erro Médio Quadrático: "+"EMQ: "+this.erro_treinamento);
		System.out.println();
		System.out.println("----------- Validação -----------");
		for (int i = 0; i < saida_desejada_validacao.length; i++) {
			System.out.println("Desejada: "+saida_desejada_validacao[i]+" Real: "+saida_real_validacao[i]);
		}
		System.out.println("Epocas: "+epoca+" Erro Médio Quadrático: "+this.erro_validacao);
	}
	
	private void teste(){
	
	this.num_padroes = this.entradas_test.length;
	//this.mlp.randomizarPesos(this.num_padroes);
	double saida_desejada_teste[] = new double[this.entradas_test.length];
	double saida_real_teste[] = new double[this.entradas_test.length];
	int epoca = 0;
	int controle = 0;
	
	while(controle < this.num_padroes){
				
		/*for (int i = 0; i < oculta.neuronios.size(); i++) {
			if(oculta.neuronios.get(i).entradas.size() == 0) oculta.neuronios.get(i).entradas.add(this.entradas_teste.get(controle));
			else oculta.neuronios.get(i).entradas.set(0, this.entradas_teste.get(controle));
		}*/
		
		for (int i = 0; i < oculta.neuronios.size(); i++) {
			if(oculta.neuronios.get(i).entradas.size() == 0){
				for (int j = 0; j < this.entradas_test[controle].length; j++) {
					oculta.neuronios.get(i).entradas.add(entradas_test[controle][j]);
				}	
				
			}
			else{
				for (int j = 0; j < this.entradas_test[controle].length; j++) {
					oculta.neuronios.get(i).entradas.set(j, entradas_test[controle][j]);
				}
				
			}
		}
			
		this.bp.setSaidaDesejada(this.saidas_teste.get(controle));
		this.bp.propagacao();
		saida_desejada_teste[controle] = this.bp.saidaDesejada;
		saida_real_teste[controle] = this.mlp.saidaReal;
		//this.bp.retropropagacao();
		//this.bp.propagacao();
			
		for (int i = 0; i < this.mlp.camadas.size(); i++) {
			this.mlp.camadas.get(i).gradiente_local.clear();
		}
		
		
				
		controle++;
	}
		controle = 0;
		epoca ++;
		
		this.erro_teste = this.bp.erro_medio_quad(/*num_padroes*/num_padroes(entradas_test));
		this.bp.somatorio_energia.clear();
		
		System.out.println();
		System.out.println("----------- Teste -----------");
		for (int i = 0; i < saida_real_teste.length; i++) {
			System.out.println("Desejada: "+saida_desejada_teste[i]+" Real: "+saida_real_teste[i]);
		}
		System.out.println("Epocas: "+epoca+" Erro Médio Quadrático: "+this.erro_teste);
		System.out.println();
		System.out.println("Término: "+sdf.format(new Date()));
	}
	
	private void preparar(){
		
		arquivo();
		
		this.entradas_treinamento = new ArrayList<>();
		this.entradas_validacao = new ArrayList<>();
		this.entradas_teste = new ArrayList<>();
		this.saidas_treinamento = new ArrayList<>();
		this.saidas_validacao = new ArrayList<>();
		this.saidas_teste = new ArrayList<>();
		this.neuronios = new ArrayList<>();
		
		this.mlp = new MLP();
		this.bp = new BackPropagation(this.mlp, this.TAXA_APREND, this.MOMENTO);
		
		this.oculta = new Camada();
		this.saida = new Camada();
		
		entradas_trein = new double[][]{
			
			{0.000, 0.001}, {0.001, 0.024}, {0.024, 0.081}, {0.081, 0.087},
			{0.087, 0.037}, {0.037, 0.011}, {0.011, 0.009}, {0.009, 0.007},
			{0.007, 0.003}, {0.003, 0.008},//2007 
			
			{0.008, 0.007}, {0.007, 0.001},
			{0.001, 0.002}, {0.002, 0.028}, {0.028, 0.161}, {0.161, 0.123},
			{0.123, 0.018}, {0.018, 0.009}, {0.009, 0.003}, {0.003, 0.002},
			{0.002, 0.005}, {0.005, 0.000},//2008 
			
			{0.000, 0.003}, {0.003, 0.013},{0.013, 0.009}, {0.009, 0.011}, 
			{0.011, 0.005}, {0.005, 0.001},{0.001, 0.000}, {0.000, 0.002}, 
			{0.002, 0.002}, {0.002, 0.009},{0.009, 0.005}, {0.005, 0.001}, //2009
		
			{0.001, 0.002}, {0.002, 0.003},{0.003, 0.003}, {0.003, 0.011},
			{0.011, 0.019}, {0.019, 0.075},{0.075, 0.135}, {0.135, 0.050}, 
			{0.050, 0.027}, {0.027, 0.008},{0.008, 0.015}, {0.015, 0.016}, //2010
		
					//{0.000, 0.001, 0.024, 0.081, 0.087, 0.037, 0.011, 0.009, 0.007, 0.003, 0.008, 0.007},//2007
					//{0.001, 0.002, 0.028, 0.161, 0.123, 0.018, 0.009, 0.003, 0.002, 0.005, 0.000, 0.003},//2008
					//{0.013, 0.009, 0.011, 0.005, 0.001, 0.000, 0.002, 0.002, 0.009, 0.005, 0.001, 0.002},//2009
					//{0.003, 0.003, 0.011, 0.019, 0.075, 0.135, 0.050, 0.027, 0.008, 0.015, 0.016, 0.056},//2010
					//{0.141, 0.231, 0.430, 0.187, 0.064, 0.034, 0.015, 0.009, 0.013, 0.021, 0.035, 0.089},//2011
					//{0.109, 0.664, 0.938, 0.589, 0.156, 0.069, 0.037, 0.007, 0.000, 0.008, 0.005, 0.006},//2012
		};
		
		entradas_val = new double[][]{
			
			{0.016, 0.056}, {0.056, 0.141}, {0.141, 0.231}, {0.231, 0.430},
			{0.430, 0.187}, {0.187, 0.064}, {0.064, 0.034}, {0.034, 0.015},
			{0.015, 0.009}, {0.009, 0.013}, {0.013, 0.021}, {0.021, 0.035},//2011
		
					
		};
		
		entradas_test = new double[][]{
			
			
			{0.035, 0.089}, {0.089, 0.109}, {0.109, 0.664}, {0.664, 0.938},
			{0.938, 0.589}, {0.589, 0.156}, {0.156, 0.069}, {0.069, 0.037},
			{0.037, 0.007}, {0.007, 0.000}, {0.000, 0.008}, {0.008, 0.005},//2012
			
		};
		
		this.saidas_treinamento.add(0.024);
		this.saidas_treinamento.add(0.081);
		this.saidas_treinamento.add(0.087);
		this.saidas_treinamento.add(0.037);
		this.saidas_treinamento.add(0.011);
		this.saidas_treinamento.add(0.009);
		this.saidas_treinamento.add(0.007);
		this.saidas_treinamento.add(0.003);
		this.saidas_treinamento.add(0.008);
		this.saidas_treinamento.add(0.007);
		
		this.saidas_treinamento.add(0.001);
		this.saidas_treinamento.add(0.002);
		this.saidas_treinamento.add(0.028);
		this.saidas_treinamento.add(0.161);
		this.saidas_treinamento.add(0.123);
		this.saidas_treinamento.add(0.018);
		this.saidas_treinamento.add(0.009);
		this.saidas_treinamento.add(0.003);
		this.saidas_treinamento.add(0.002);
		this.saidas_treinamento.add(0.005);
		this.saidas_treinamento.add(0.000);
		this.saidas_treinamento.add(0.003);
		
		this.saidas_treinamento.add(0.013);
		this.saidas_treinamento.add(0.009);
		this.saidas_treinamento.add(0.011);
		this.saidas_treinamento.add(0.005);
		this.saidas_treinamento.add(0.001);
		this.saidas_treinamento.add(0.000);
		this.saidas_treinamento.add(0.002);
		this.saidas_treinamento.add(0.002);
		this.saidas_treinamento.add(0.009);
		this.saidas_treinamento.add(0.005);
		this.saidas_treinamento.add(0.001);
		this.saidas_treinamento.add(0.002);
		
		this.saidas_treinamento.add(0.003);
		this.saidas_treinamento.add(0.003);
		this.saidas_treinamento.add(0.011);
		this.saidas_treinamento.add(0.019);
		this.saidas_treinamento.add(0.075);
		this.saidas_treinamento.add(0.135);
		this.saidas_treinamento.add(0.050);
		this.saidas_treinamento.add(0.027);
		this.saidas_treinamento.add(0.008);
		this.saidas_treinamento.add(0.015);
		this.saidas_treinamento.add(0.016);
		this.saidas_treinamento.add(0.056);
		
		this.saidas_validacao.add(0.141);
		this.saidas_validacao.add(0.231);
		this.saidas_validacao.add(0.430);
		this.saidas_validacao.add(0.187);
		this.saidas_validacao.add(0.064);
		this.saidas_validacao.add(0.034);
		this.saidas_validacao.add(0.015);
		this.saidas_validacao.add(0.009);
		this.saidas_validacao.add(0.013);
		this.saidas_validacao.add(0.021);
		this.saidas_validacao.add(0.035);
		this.saidas_validacao.add(0.089);
		
		this.saidas_teste.add(0.109);
		this.saidas_teste.add(0.664);
		this.saidas_teste.add(0.938);
		this.saidas_teste.add(0.589);
		this.saidas_teste.add(0.156);
		this.saidas_teste.add(0.069);
		this.saidas_teste.add(0.037);
		this.saidas_teste.add(0.007);
		this.saidas_teste.add(0.000);
		this.saidas_teste.add(0.008);
		this.saidas_teste.add(0.005);
		this.saidas_teste.add(0.006);
		
	
		
		/****** VALIDAÇÃO ******/
		/*this.entradas_treinamento.add(0.000);// 0.0
		this.entradas_treinamento.add(0.001);// 1.0
		this.entradas_treinamento.add(0.002);// 2.0
		this.entradas_treinamento.add(0.003);// 3.0
		this.entradas_treinamento.add(0.004);
		this.entradas_treinamento.add(0.005);
		this.entradas_validacao.add(0.006);// 4.0
		this.entradas_validacao.add(0.007);// 5.0
		this.entradas_teste.add(0.008);// 8.0
		this.entradas_teste.add(0.009);// 9.0*/
		
		/*this.saidas_treinamento.add(0.003);// 0.003
		this.saidas_treinamento.add(0.004);// 0.004
		this.saidas_treinamento.add(0.007);// 0.007
		this.saidas_treinamento.add(0.012);// 0.012
		this.saidas_treinamento.add(0.019);// 0.019
		this.saidas_treinamento.add(0.028);// 0.028
		this.saidas_validacao.add(0.039);// 0.039
		this.saidas_validacao.add(0.052);// 0.052
		this.saidas_teste.add(0.067);// 0.067
		this.saidas_teste.add(0.084);// 0.084*/
		/****** VALIDAÇÃO ******/
		
		Neuronio n;
		//Neuronio_Saida ns = new Neuronio_Saida();
		
		for (int i = 0; i < 11; i++) {
			n = new Neuronio();
			this.neuronios.add(n);
		}
		
		for (int i = 0; i < neuronios.size(); i++) {
			if(i == neuronios.size()-1) this.saida.neuronios.add(this.neuronios.get(i));
			else /*if(i < neuronios.size()/2)*/this.oculta.neuronios.add(this.neuronios.get(i));
					// else this.oculta2.neuronios.add(this.neuronios.get(i));
		}
		
		//this.saida.neuronios.add(ns);
		
		this.mlp.addCamada(this.oculta);
		this.mlp.addCamada(this.saida);
	}
	
	private void informacoes(){
		
		System.out.println();
		System.out.println("---------------------- Informações ----------------------");
		System.out.println("Rede Neural(MLP) - Taxa de Aprendizagem: "+this.TAXA_APREND+" Momento: "+this.MOMENTO);
		System.out.println("Qtd. de Neurônios na Camada Oculta: "+this.mlp.camadas.get(0).neuronios.size());
		System.out.println();
		System.out.println("---------- Pesos por Neurônio da Camada Oculta ----------");
		System.out.println();
		for (int i = 0; i < this.mlp.camadas.get(0).neuronios.size(); i++){
			System.out.print("Neurônio: "+i+"\n");
			for (int j = 0; j < this.mlp.camadas.get(0).neuronios.get(i).pesos.size(); j++) {
				System.out.println("\tPeso: "+this.mlp.camadas.get(0).neuronios.get(i).pesos.get(j));
			}
		}
		//System.out.println("---------- Pesos por Neurônio da Camada de Saída ----------");
		//System.out.println("Neurônio: 0 \tPeso:"+this.mlp.camadas.get(this.mlp.camadas.size()-1).neuronios.get(0).pesos.get(0));
	}
	
	private void cabecalho(){
		
		System.out.println("-----------------------------------");
		System.out.println("Universidade Federal Rural de Pernambuco(UFRPE)");
		System.out.println("Unidade Acadêmica de Serra Talhada(UAST)");
		System.out.println("Curso: Bacharelado em Sistemas de Informação");
		System.out.println("Trabalho de Conclusão de Curso");
		System.out.println("Aluno: Renê Douglas Nobre de Morais");
		System.out.println("Orientador: Glauber Magalhães Pires");
		System.out.println("-----------------------------------");
		System.out.println();
		System.out.println("Teste de Validação da Rede Neural Artificial - Multilayer Perceptrons (MLP)");
		System.out.println();
		
	}
	
	private void arquivo(){
		
		try {
			this.arq = new FileWriter("grafico.csv");
			this.gravar = new PrintWriter(arq);
			this.gravar.printf("epoca; erro%n");
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	private int num_padroes(double matriz[][]){
		
		int count = 0;
		
		for (int i = 0; i < matriz.length; i++) {
			for (int j = 0; j < matriz[i].length; j++) {
				count ++;
			}
		}
		return count;
	}
	
	public static void main(String[] args) {
		
		new Treinar();
		
	}

}
