package mlp;

import java.util.ArrayList;
import java.util.Random;

public class MLP {
	
	public ArrayList<Camada> camadas;
	public double saidaReal;
	//public double saidaReal2;
	
	public MLP() {
		this.camadas = new ArrayList<Camada>();
	}
	
	public void addCamada(Camada camada){
		
		this.camadas.add(camada);
	}
	
	public void randomizarPesos(int num_padroes, int num_entradas){
		
		int num_pesos = 0;
		double valor = 0.0;
		int num_pesos_negativos = 0;
		double peso;
		Random r = new Random();
		
		for (int i = 0; i < camadas.size(); i++) {
				if(i == 0){
					num_pesos += camadas.get(i).neuronios.size();
				}
				else{
					num_pesos += (camadas.get(i-1).neuronios.size()*camadas.get(i).neuronios.size());
				}
		}
		num_pesos_negativos = num_pesos/2;
		valor = 1/Math.sqrt(num_padroes);
		
		for (int i = 0; i < camadas.size(); i++) {
			for (int j = 0; j < camadas.get(i).neuronios.size(); j++) {
				
				if(i == 0){
					
					for (int j2 = 0; j2 < num_entradas; j2++) {
						
						peso = Math.random()*valor;/*(valor-(-valor))+(-valor);*/
						camadas.get(i).neuronios.get(j).pesos.add(peso);
						camadas.get(i).neuronios.get(j).deltaAntigo.add(0.0);
					}
					
				}
				else{
					for (int j2 = 0; j2 < camadas.get(i-1).neuronios.size(); j2++) {
						
						peso = Math.random()*valor;
						camadas.get(i).neuronios.get(j).pesos.add(peso);
						camadas.get(i).neuronios.get(j).deltaAntigo.add(0.0);
					}
					peso = 0.0;
				}
			}
		}
		while(num_pesos_negativos > 0){
			for (int i = 0; i < camadas.size(); i++) {
				for (int j = 0; j < camadas.get(i).neuronios.size(); j++) {
					for (int h = 0; h < camadas.get(i).neuronios.get(j).pesos.size(); h++) {
						
							if(r.nextBoolean() == false && num_pesos_negativos > 0){
								camadas.get(i).neuronios.get(j).pesos.set(h,
										camadas.get(i).neuronios.get(j).pesos.get(h)*(-1));
								num_pesos_negativos--;
							}
					}
				}
			}
		}
		
	}
}
