<script>
	import tailwindConfig from '../../../tailwind.config';
	import resolveConfig from 'tailwindcss/resolveConfig';
	import { base } from '$app/paths';

	// import Youtube from './Youtube.svelte';

	let softmaxEquation = `$$\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}$$`;
	let reluEquation = `$$\\text{ReLU}(x) = \\max(0,x)$$`;

	let currentPlayer;

	const { theme } = resolveConfig(tailwindConfig);
</script>

<div id="description">
	<div class="article-section">
		<h1>Was ist ein Transformer?</h1>

		<p>
			Der Transformer ist eine neuronale Netzwerkarchitektur, die die Herangehensweise an Künstliche Intelligenz grundlegend verändert hat. 
			Der Transformer wurde erstmals in der bahnbrechenden Arbeit
			<a
				href="https://dl.acm.org/doi/10.5555/3295222.3295349"
				title="ACM Digital Library"
				target="_blank">"Attention is All You Need"</a
			>
			im Jahr 2017 vorgestellt und hat sich seitdem zur bevorzugten Architektur für Deep Learning-Modelle entwickelt.
			Er treibt textgenerative Modelle wie OpenAIs <strong>GPT</strong>, Metas <strong>Llama</strong> 
			und Googles <strong>Gemini</strong> an. Über den Text hinaus wird der Transformer auch in der
			<a
				href="https://huggingface.co/learn/audio-course/en/chapter3/introduction"
				title="Hugging Face"
				target="_blank">Audioerzeugung</a
			>,
			<a
				href="https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification"
				title="Hugging Face"
				target="_blank">Bilderkennung</a
			>,
			<a href="https://elifesciences.org/articles/82819" title="eLife"
				>Proteinstrukturvorhersage</a
			> und sogar beim
			<a
				href="https://www.deeplearning.ai/the-batch/reinforcement-learning-plus-transformers-equals-efficiency/"
				title="Deep Learning AI"
				target="_blank">Spielen von Spielen</a
			> eingesetzt, was seine Vielseitigkeit in verschiedenen Bereichen demonstriert. 
		</p>
		<p>
			Im Wesentlichen basieren textgenerative Transformer-Modelle auf dem Prinzip
			der <strong>Wortvorhersage</strong>: Bei einer Texteingabe durch den Benutzer, 
			welches ist das <em>wahrscheinlichste nächste Wort</em>, das auf diese Eingabe folgt? 
			Die Kerninnovation und Stärke der Transformer liegt in ihrem Einsatz des Self-Attention-Mechanismus, 
			der es ihnen ermöglicht, ganze Sequenzen zu verarbeiten und 
			langfristige Abhängigkeiten (context) effektiver zu erfassen als bisherige Architekturen. 
		</p>
		<p>
			Die GPT-2-Familie von Modellen sind prominente Beispiele für textgenerative Transformer. 
			Der Transformer Explainer wird durch 
			<a href="https://huggingface.co/openai-community/gpt2" title="Hugging Face" target="_blank"
				>GPT-2</a
			>
			(small) betrieben, ein "kleines" Modell, das 124 Millionen Parameter umfasst. 
			Obwohl es nicht das neueste oder leistungsfähigste Transformer-Modell ist, 
			teilt es viele der gleichen architektonischen Komponenten und Prinzipien, 
			die in den aktuellen Spitzenmodellen zu finden sind, was es zu einem idealen Ausgangspunkt 
			für das Verständnis der Grundlagen macht.
		</p>
	</div>

	<div class="article-section">
		<h1>Die Transformer-Architektur</h1>

		<p>
			Jeder textgenerative Transformer besteht aus diesen <strong>drei Hauptkomponenten</strong>:

		</p>
		<ol>
			<li>
				<strong class="bold-purple">Einbettung / Embedding</strong>: Die texteingabe wird in kleinere Einheiten aufgeteilt, 
				die als Tokens bezeichnet werden und Wörter oder Teilwörter sein können. Diese Tokens werden in numerische 
				Vektoren umgewandelt, die als Einbettungen bezeichnet werden und die semantische Bedeutung von Wörtern erfassen.
			</li>
			<li>
				<strong class="bold-purple">Transformer Block</strong> ist der grundlegende Baustein des Modells, 
				der die Eingabedaten verarbeitet und transformiert. Jeder Block umfasst:
				<ul class="">
					<li>
						<strong>Aufmerksamkeitsmechanismus / Attention Mechanism</strong>, das Kernstück des Transformer-Blocks. 
						Er ermöglicht es Tokens, miteinander zu kommunizieren und kontextuelle Informationen 
						sowie Beziehungen zwischen Wörtern zu erfassen.
					</li>
					<li>
						<strong>MLP (Multilayer Perceptron) Layer</strong>, ein Feed-Forward-Netzwerk, 
						das unabhängig auf jedes Token angewendet wird. Während das Ziel der Aufmerksamkeitsmechanismus-Schicht 
						darin besteht, Informationen zwischen Tokens weiterzuleiten, 
						zielt der MLP darauf ab, die Repräsentation jedes Tokens zu verfeinern.

					</li>
				</ul>
			</li>
			<li>
				<strong class="bold-purple">Ausgabe-Wahrscheinlichkeiten</strong>: Die finalen linearen und Softmax-Schichten transformieren 
				die verarbeiteten Einbettungen in Wahrscheinlichkeiten, wodurch das Modell Vorhersagen über das 
				nächste Token in einer Sequenz treffen kann.

			</li>
		</ol>

		<div class="architecture-section" id="embedding">
			<h2>Einbettung / Embedding</h2>
			<p>
				Angenommen, Sie möchten Text mit einem Transformer-Modell generieren. Sie geben die Aufforderung wie diese ein: 
				<code>“Data visualization empowers users to”</code>. Diese Eingabe muss in ein Format umgewandelt werden, 
				das das Modell verstehen und verarbeiten kann. Hier kommt die Einbettung / Embedding ins Spiel: 
				Sie verwandelt den Text in eine numerische Repräsentation, mit der das Modell arbeiten kann. 
				Um eine Eingabeaufforderung in eine Einbettung umzuwandeln, müssen wir 1) den Eingabetext tokenisieren, 
				2) Token-Einbettungen erhalten, 3) Positionsinformationen hinzufügen und schließlich 
				4) Token- und Positionscodierungen zusammenfügen, um die endgültige Einbettung zu erhalten. 
				Lassen Sie uns betrachten, wie jeder dieser Schritte durchgeführt wird.

			</p>
			<div class="figure">
				<img src="./article_assets/embedding.png" width="60%" height="60%" align="middle" />
			</div>
			<div class="figure-caption">
				Abbildung <span class="attention">1</span>. Erweiterung der Ansicht der Einbettungsschicht / des Embedding Layer, 
				die zeigt, wie die Eingabeaufforderung in eine Vektorrepräsentation umgewandelt wird. Der Prozess umfasst 
				<span class="fig-numbering">(1)</span> Tokenisierung / Tokenization, (2) Token-Einbettung / Token Embedding, 
				(3) Positionscodierung / Positional Encoding und (4) Finale Einbettung / Finale Embedding 

			</div>
			<div class="article-subsection">
				<h3>Schritt 1: Tokenisierung / Tokenization</h3>
				<p>
					Die Tokenisierung ist der Prozess des Zerlegens des Eingabetexts in kleinere, handlichere Stücke, 
					die als Tokens bezeichnet werden. Diese Tokens können ein Wort oder ein Teilwort sein. 
					Die Wörter <code>"Data"</code> und <code>"visualization"</code> entsprechen eindeutigen Tokens, 
					während das Wort <code>"empowers"</code> in zwei Tokens aufgeteilt wird. Das vollständige Vokabular der 
					Tokens wird vor dem Training des Modells festgelegt: Das Vokabular von GPT-2 umfasst <code>50.257</code> 
					eindeutige Tokens. Jetzt, da wir unseren Eingabetext in Tokens mit eindeutigen IDs aufgeteilt haben, 
					können wir ihre Vektorrepräsentation aus den Einbettungen erhalten.
				</p>
			</div>
			<div class="article-subsection" id="article-token-embedding">
				<h3>Schritt 2: Token-Einbettung</h3>
				<p>
				 GPT-2 Small stellt jedes Token im Vokabular als einen 768-dimensionalen Vektor dar; 
				die Dimension des Vektors hängt vom Modell ab. Diese Einbettungsvektoren werden in einer Matrix 
    				mit der Form <code>(50.257, 768)</code> gespeichert, die etwa 39 Millionen Parameter enthält! 
				Diese umfangreiche Matrix ermöglicht es dem Modell, jedem Token eine semantische Bedeutung zuzuweisen.
				</p>
			</div>
			<div class="article-subsection" id="article-positional-embedding">
				<h3>Schritt 3: Positionscodierung / Positional Encoding</h3>
				<p>
				Die Einbettungsschicht / Embedding Layer kodiert auch Informationen über die Position jedes Tokens der Text Inputs.
				Verschiedene Modelle verwenden unterschiedliche Methoden zur Positionscodierung. GPT-2 trainiert seine eigene Positionscodierungsmatrix
				von Grund auf und integriert sie direkt in den Trainingsprozess.
				</p>

				<!-- <div class="article-subsection-l2">
            <h4>Alternative Positionskodierungsansatz / Alternative Positional <strong class='attention'>[POTENZIELL ZUSAMMENKLAPPBAR]</strong></h4>
            <p>
              Andere Modelle, wie der originale Transformer und BERT, verwenden sinusförmige Funktionen für die Positionskodierung.

              Diese sinusförmige Kodierung ist deterministisch und darauf ausgelegt,
              sowohl die absolute als auch die relative Position jedes Tokens widerzuspiegeln.
            </p>
            <p>
              	Jede Position in einer Sequenz wird durch eine einzigartige mathematische Repräsentation 
		unter Verwendung einer Kombination aus Sinus- und Kosinusfunktionen zugewiesen.

              	Für eine gegebene Position repräsentiert die Sinusfunktion die geraden Dimensionen,
          	und die Kosinusfunktion repräsentiert die ungeraden Dimensionen innerhalb des Positionskodierungsvektors.

          	Diese periodische Natur stellt sicher, dass jede Position eine konsistente Kodierung erhält,
          	unabhängig vom umgebenden Kontext.
            </p>

            <p>
             So funktioniert es:
            </p>

            <span class='attention'>
          SINUSFÖRMIGE POSITIONSKODIERUNGSGLEICHUNG
        </span>

        <ul>
          <li>
            <strong>Sinusfunktion</strong>: Wird für gerade Indizes des Einbettungsvektors verwendet.
          </li>
          <li>
            <strong>Kosinusfunktion</strong>: Wird für ungerade Indizes des Einbettungsvektors verwendet.
        </ul>

        <p>
          Fahren Sie mit der Maus über einzelne Kodierungswerte in der obigen Matrix,
          um zu sehen, wie diese mit den Sinus- und Kosinusfunktionen berechnet werden.
        </p>
          </div> -->
			</div>
			<div class="article-subsection">
				<h3>Schritt 4. Finale Einbettung / Final Embedding </h3>
				<p>
					Abschließend summieren wir die Token- und Positionskodierungen, um die finale Repräsentation der Einbettung / finale embedding zu erhalten. 
					Diese kombinierte Repräsentation erfasst sowohl die semantische Bedeutung 
					der Tokens als auch ihre Position in der Eingabesequenz.
				</p>
			</div>
		</div>

		<div class="architecture-section">
			<h2>Transformer Block</h2>

			<p>	Der Kern der Verarbeitung im Transformer liegt im Transformer-Block, 
				der aus einer Multi-Head-Selbstaufmerksamkeit (multi-head self-attention) und einer Multi-Layer-Perceptron-Schicht besteht. 
				Die meisten Modelle bestehen aus mehreren solcher Blöcke, die nacheinander sequenziell gestapelt sind. 
				Die Token-Repräsentationen entwickeln sich durch die Schichten, vom ersten Block bis zum zwölften, 
				was dem Modell ermöglicht, ein komplexes Verständnis für jedes Token aufzubauen. 
				Dieser geschichtete Ansatz führt zu höherwertigen Repräsentationen der Eingabe.
			</p>

			<div class="article-subsection" id="self-attention">
				<h3>Multi-Head Selbstaufmerksamkeit / Multi-Head Self-Attention</h3>
				<p>	
					Der Selbstaufmerksamkeitsmechanismus /  self-attention mechanism  ermöglicht es dem Modell, 
					sich auf relevante Teile der Eingabesequenz zu konzentrieren und so komplexe Beziehungen und Abhängigkeiten 
					innerhalb der Daten zu erfassen. Schauen wir uns an, wie diese Selbstaufmerksamkeit / self attention Schritt 
					für Schritt berechnet wird.
				</p>
				<div class="article-subsection-l2">
					<h4>Schritt 1: Query, Key, und Value Matrizen</h4>

					<div class="figure">
						<img src="./article_assets/QKV.png" width="80%" align="middle" />
					</div>
					<div class="figure-caption">
						Abbildung <span class="attention">2</span>.Berechnung der Query-, Key- und Value-Matrizen 
						aus der ursprünglichen Einbettung.
					</div>

					<p>
						Each token's embedding vector is transformed into three vectors:
						<span class="q-color">Query (Q)</span>,
						<span class="k-color">Key (K)</span>, und
						<span class="v-color">Value (V)</span>. Diese Vektoren werden abgeleitet, 
						indem die Eingabe-Einbettungsmatrix mit gelernten Gewichtsmatrizen für
						<span class="q-color">Q</span>,
						<span class="k-color">K</span>, und
						<span class="v-color">V</span> multipliziert wird. Hier ist eine Analogie zur Websuche, 
						um ein intuitives Verständnis für diese Matrizen zu entwickeln:
					
					</p>
					<ul>
						<li>
							<strong class="q-color font-medium">Query (Q)</strong> ist der Suchtext, 
							den Sie in die Suchleiste einer Suchmaschine eingeben. Dies ist das Token, 
							über das Sie <em>"mehr Informationen finden möchten"</em>.
						</li>
						<li>
							<strong class="k-color font-medium">Key (K)</strong> ist der Titel jeder Webseite 
							im Suchergebnisfenster. Er repräsentiert die möglichen Tokens, auf die sich die Query konzentrieren kann.
						</li>
						<li>
							<strong class="v-color font-medium">Value (V)</strong> ist der tatsächliche Inhalt der 
							angezeigten Webseiten. Nachdem wir den passenden Suchbegriff (Query) mit den relevanten 
							Ergebnissen (Key) abgeglichen haben, möchten wir den Inhalt (Value) der relevantesten Seiten erhalten.
						</li>
					</ul>
					<p>
						Durch die Verwendung dieser QKV-Werte kann das Modell Aufmerksamkeitswerte (attention scores) berechnen, die bestimmen, 
						wie viel Fokus jedes Token bei der Generierung von Vorhersagen erhalten sollte.
					</p>
				</div>
				<div class="article-subsection-l2">
					<h4>Schritt 2: Maskierte Selbstaufmerksamkeit / Masked Self-Attention</h4>
					<p>
						Die maskierte Selbstaufmerksamkeit / Masked self-attention ermöglicht es dem Modell, Sequenzen zu generieren, 
						indem es sich auf relevante Teile der Eingabe konzentriert und gleichzeitig den Zugriff auf zukünftige Tokens
						verhindert. </p>

					<div class="figure">
						<img src="./article_assets/attention.png" width="80%" align="middle" />
					</div>
					<div class="figure-caption">
						Abbildung <span class="attention">3</span>. Verwendung von Query-, Key- und Value-Matrizen zur Berechnung 
						der maskierten Selbstaufmerksamkeit / Masked Self-Attention.
					</div>

					<ul>
						<li>
							<strong>Aufmerksamkeitswert / Attention Score</strong>: Das Skalarprodukt der (dot product) der
							<span class="q-color">Query</span>
							und <span class="k-color">Key</span> -Matrizen bestimmt die Ausrichtung jeder Query mit jedem Key und erzeugt 
							eine quadratische Matrix, die die Beziehung zwischen allen Eingabetokens widerspiegelt.
						</li>
						<li>
							<strong>Maskierung</strong>: Eine Maske wird auf das obere Dreieck der Aufmerksamkeitsmatrix angewendet, 
							um zu verhindern, dass das Modell auf zukünftige Tokens zugreift, indem diese Werte auf 
							„negative Unendlichkeit" gesetzt werden. Das Modell muss lernen, den nächsten Token vorherzusagen, ohne in die Zukunft „zu schauen".
						</li>
						<li>
							<strong>Softmax</strong>: Nach der Maskierung wird der Aufmerksamkeitswert durch die Softmax-Operation 
							in eine Wahrscheinlichkeit umgewandelt, bei der das Exponential jeder Aufmerksamkeitsbewertung (attention score) genommen wird. 
							Jede Zeile der Matrix summiert sich zu eins und zeigt die Relevanz jedes anderen Tokens links davon an.
			
						</li>
					</ul>
				</div>
				<div class="article-subsection-l2">
					<h4>Schritt 3: Ausgabe / Output</h4>
					<p>
						Das Modell verwendet die maskierten Selbstaufmerksamkeitswerte / masked self-attention und multipliziert sie mit 
						der <span class="v-color">Value</span>-Matrix, um die <span class="purple-color">fianle Ausgabe</span> 
						des Selbstaufmerksamkeitsmechanismus zu erhalten. GPT-2 verfügt über <code>12</code> Selbstaufmerksamkeitsköpfe / self-attention heads, 
						die jeweils unterschiedliche Beziehungen zwischen den Tokens erfassen. Die Ausgaben dieser Köpfe werden 
						zusammengeführt und durch eine lineare Projektion weiterverarbeitet.

					</p>F
				</div>

				<div class="article-subsection" id="article-activation">
					<h3>MLP: Multi-Layer Perceptron (Mehrschichtiger Perzeptron) </h3>

					<div class="figure">
						<img src="./article_assets/mlp.png" width="70%" align="middle" />
					</div>
					<div class="figure-caption">
						Abbildung  <span class="attention">4</span>. Verwendung der MLP-Schicht, um die Repräsentation der Selbstaufmerksamkeit 
						in höhere Dimensionen zu projizieren und so die Repräsentationskapazität des Modells zu erhöhen.
					</div>

					<p>	
						Nachdem die verschiedenen Köpfe der Selbstaufmerksamkeit / self-attention die vielfältigen Beziehungen zwischen 
						den Eingabetokens erfasst haben, werden die zusammengeführten Ausgaben durch die Multi-Layer Perceptron-(MLP)-Schicht 
						geleitet, um die Repräsentationskapazität des Modells zu erhöhen. Der MLP-Block besteht aus zwei linearen 
						Transformationen mit einer GELU-Aktivierungsfunktion dazwischen. Die erste lineare Transformation erhöht 
						die Dimensionalität der Eingabe um das Vierfache von <code>768</code> auf <code>3072</code>. Die zweite lineare 
						Transformation reduziert die Dimensionalität wieder auf die ursprüngliche Größe von <code>768</code>, 
						wodurch sichergestellt wird, dass die nachfolgenden Schichten Eingaben mit konsistenter Dimensionalität erhalten. 
						Im Gegensatz zum Selbstaufmerksamkeitsmechanismus / self-attention mechanism verarbeitet der MLP Tokens unabhängig voneinander und mappt sie 
						einfach von einer Repräsentation zur nächsten.

						
					</p>
				</div>

				<div class="architecture-section" id="article-prob">
					<h2>Ausgabe-Wahrscheinlichkeiten / Output Probabilities</h2>
					<p>
						Nachdem die Eingabe durch alle Transformer-Blöcke verarbeitet wurde, wird die Ausgabe durch die 
						abschließende lineare Schicht geleitet, um sie für die Token-Vorhersage vorzubereiten. Diese Schicht projiziert 
						die endgültigen Repräsentationen in einen <code>50.257</code>-dimensionalen Raum, in dem jedes Token im Vokabular 
						einen entsprechenden Wert namens <code>Logit</code> hat. Da jedes Token das nächste Wort sein kann, 
						ermöglicht uns dieser Prozess, diese Tokens einfach nach ihrer Wahrscheinlichkeit, das nächste Wort zu sein, 
						zu ordnen. Anschließend wenden wir die Softmax-Funktion an, um die Logits in eine Wahrscheinlichkeitsverteilung 
						umzuwandeln, die sich auf eins summiert. Dadurch können wir das nächste Token basierend auf 
						seiner Wahrscheinlichkeit auswählen.
					</p>

					<div class="figure">
						<img src="./article_assets/softmax.png" width="60%" align="middle" />
					</div>
					<div class="figure-caption">
						Abbildung <span class="attention">5</span>. Jedem Token im Vokabular wird basierend auf den Logits 
						des Modells eine Wahrscheinlichkeit zugewiesen. Diese Wahrscheinlichkeiten bestimmen die Wahrscheinlichkeit,
						dass jedes Token das nächste Wort in der Sequenz wird.
					</div>

					<p id="article-temperature">

						Der letzte Schritt besteht darin, das nächste Token zu generieren, indem aus dieser Verteilung eine Auswahl 
						getroffen wird. Der <code>Temperatur</code>-Hyperparameter spielt in diesem Prozess eine entscheidende Rolle. 
						Mathematisch gesehen ist dies ein sehr einfacher Vorgang: Die Logits des Modellausgangs werden einfach durch 
						die <code>Temperatur</code> dividiert:

					</p>

					<ul>
						<li>
							<code>Temperatur = 1</code>: Die Division der Logits durch eins hat keine Auswirkung auf die Softmax-Ausgaben.
						</li>
						<li>
							<code>Temperatur &lt; 1</code>: Eine niedrigere Temperatur macht das Modell selbstbewusster und deterministischer, 
							indem sie die Wahrscheinlichkeitsverteilung schärft, was zu vorhersehbareren Ausgaben führt.
						</li>
						<li>
							<code>Temperatur &gt; 1</code>: Eine höhere Temperatur erzeugt eine weichere Wahrscheinlichkeitsverteilung, 
							was mehr Zufälligkeit im generierten Text zulässt – was manche als <em>„Kreativität“</em> des Modells bezeichnen.
						</li>
					</ul>


					<p>
						Stellen Sie die Temperatur ein und sehen Sie, wie Sie das Gleichgewicht zwischen deterministischen und 
						„kreativen“ Ausgaben finden können!

					</p>
				</div>

				<div class="architecture-section">
					<h2>Erweiterte Architekturmerkmale</h2>

					<p>
						Es gibt mehrere erweiterte Architekturmerkmale, die die Leistung von Transformer-Modellen verbessern. 
						Obwohl sie für die Gesamtleistung des Modells wichtig sind, sind sie nicht so entscheidend für das Verständnis 
						der Kernkonzepte der Architektur. Layer-Normalisierung, Dropout und Residualverbindungen sind entscheidende 
						Komponenten in Transformer-Modellen, insbesondere während der Trainingsphase. 
						Die Layer-Normalisierung stabilisiert das Training und hilft dem Modell, schneller zu konvergieren. 
						Dropout verhindert "overfitting", also das Phänomen, bei dem ein Modell die Trainingsdaten zu genau lernt und 
						dadurch auf neuen, ungesehenen Daten schlecht generalisiert, indem es zufällig Neuronen deaktiviert. 
						Residualverbindungen ermöglichen es den Gradienten, die mathematischen Größen, 
						die zur Anpassung der Modellparamter dienen, direkt durch das Netzwerk zu fließen und verhindern somit das Problem, 
						dass sie in tiefen Schichten zu klein werden, was das Lernen behindern würde.

						
					</p>
					<div class="article-subsection" id="article-ln">
						<h3>Layer-Normalisierung</h3>

						<p>
							Die Layer-Normalisierung hilft, den Trainingsprozess zu stabilisieren und die Konvergenz zu verbessern. 
							Sie funktioniert, indem sie die Eingaben über die Merkmale hinweg normalisiert und sicherstellt, 
							dass der Mittelwert und die Varianz der Aktivierungen konsistent sind. Diese Normalisierung hilft, 
							Probleme im Zusammenhang mit dem internen Kovariaten-Shift zu mindern, wodurch das Modell effektiver lernen 
							kann und weniger empfindlich auf die anfänglichen Gewichte / Parametrisierung reagiert. 
							Die Layer-Normalisierung wird in jedem Transformer-Block zweimal angewendet: einmal vor dem 
							Selbstaufmerksamkeitsmechanismus / self-attention mechanism und einmal vor der MLP-Schicht.

						</p>
					</div>
					<div class="article-subsection" id="article-dropout">
						<h3>Dropout</h3>

						<p>
							Dropout ist eine Regularisierungstechnik, die eingesetzt wird, um Overfitting in neuronalen Netzen zu verhindern, 
							indem während des Trainings zufällig ein Teil der Modellgewichte auf null gesetzt wird. 
							Dies zwingt das Modell, robustere Merkmale zu lernen und verringert die Abhängigkeit von bestimmten Neuronen, 
							wodurch das Netzwerk besser auf neue, ungesehene Daten generalisieren kann. Während der Modellinferenz 
							wird Dropout deaktiviert, was im Wesentlichen bedeutet, dass wir ein Ensemble der trainierten Unternetze 
							verwenden, was zu einer besseren Modellleistung führt.
						</p>
					</div>
					<div class="article-subsection" id="article-residual">
						<h3>Residualverbindungen</h3>

						<p>
							Residualverbindungen wurden erstmals 2015 im ResNet-Modell eingeführt und stellten eine 
							bahnbrechende Innovation im Deep-Learning dar, da sie das Training sehr tiefer neuronaler Netze ermöglichten.
							Im Wesentlichen sind Residualverbindungen Abkürzungen, die eine oder mehrere Schichten umgehen, 
							indem sie die Eingabe einer Schicht zu deren Ausgabe hinzufügen. Dies trägt dazu bei, 
							das Problem des verschwindenden / vanishing Gradienten zu mildern, wodurch es einfacher wird, 
							tiefe Netzwerke mit mehreren übereinander gestapelten Transformer-Blöcken zu trainieren. 
							In GPT-2 werden Residualverbindungen in jedem Transformer-Block zweimal verwendet: einmal vor der MLP-Schicht
							und einmal danach, um sicherzustellen, dass die Gradienten leichter fließen und die früheren Schichten 
							während der Backpropagation ausreichend aktualisiert werden.
						</p>
					</div>
				</div>

				<div class="article-section">
					<h1>Interaktive Funktionen</h1>
					<p>	
						Der Transformer-Explainer wurde interaktiv gestaltet und ermöglicht es Ihnen, die inneren Abläufe des Transformers
						zu erkunden. Hier sind einige der interaktiven Funktionen, mit denen Sie experimentieren können:
					</p>

					<ul>
						<li>
							<strong>Geben Sie Ihre eigene Textsequenz ein</strong>, um zu sehen, wie das Modell sie verarbeitet 
							und das nächste Wort vorhersagt. Erkunden Sie die Aufmerksamkeitsgewichte / attention weights, Zwischenberechnungen und sehen Sie,
							wie die finalen Ausgabe-Wahrscheinlichkeiten berechnet werden.
						</li>
						<li>
							<strong>Verwenden Sie den Temperaturregler</strong>, um die Zufälligkeit der Vorhersagen des Modells zu steuern.
							Erkunden Sie, wie Sie das Modell durch Änderung des Temperaturwerts deterministischer oder kreativer machen können.
						</li>
						<li>
							<strong>Interagieren Sie mit den Aufmerksamkeitskarten / attention maps</strong>, um zu sehen, wie das Modell 
							sich auf verschiedene Tokens in der Eingabesequenz konzentriert. Bewegen Sie den Mauszeiger über Tokens, 
							um ihre Aufmerksamkeitsgewichte / attention weights hervorzuheben, und erkunden Sie, 
							wie das Modell Kontext und Beziehungen zwischen Wörtern erfasst.
						</li>
					</ul>
				</div>

				<div class="article-section">
					<h2>Video Anleitung</h2>
					<div class="video-container">
						<iframe
							src="https://www.youtube.com/embed/ECR4oAwocjs"
							frameborder="0"
							allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
							allowfullscreen
						>
						</iframe>
					</div>
				</div>

				<div class="article-section">
					<h2>Wie ist der Transformer Explainer implementiert?</h2>
					<p>
						Der Transformer Explainer verwendet ein live laufendes GPT-2 (small) Modell direkt im Browser. 
						Dieses Modell stammt aus der PyTorch-Implementierung von GPT durch Andrej Karpathys 
						<a href="https://github.com/karpathy/nanoGPT" title="Github" target="_blank">nanoGPT-Projekt</a> 
						und wurde in <a href="https://onnxruntime.ai/" title="ONNX" target="_blank">ONNX Runtime</a> umgewandelt, 
						um eine nahtlose Ausführung im Browser zu ermöglichen. Die Benutzeroberfläche ist mit JavaScript erstellt, 
						wobei <a href="https://kit.svelte.dev/" title="Svelte" target="_blank">Svelte</a> als Front-End-Framework und 
						<a href="http://D3.js" title="D3" target="_blank">D3.js</a> für die Erstellung dynamischer Visualisierungen 
						verwendet wird. Numerische Werte werden in Echtzeit basierend auf den Benutzereingaben aktualisiert.
					</p>
				</div>

				<div class="article-section">
					<h2>Wer hat den Transformer Explainer entwickelt?</h2>
					<p>
						Der Transformer Explainer wurde von

						<a href="https://aereeeee.github.io/" target="_blank">Aeree Cho</a>,
						<a href="https://www.linkedin.com/in/chaeyeonggracekim/" target="_blank">Grace C. Kim</a>,
						<a href="https://alexkarpekov.com/" target="_blank">Alexander Karpekov</a>,
						<a href="https://alechelbling.com/" target="_blank">Alec Helbling</a>,
						<a href="https://zijie.wang/" target="_blank">Jay Wang</a>,
						<a href="https://seongmin.xyz/" target="_blank">Seongmin Lee</a>,
						<a href="https://bhoov.com/" target="_blank">Benjamin Hoover</a> und
						<a href="https://poloclub.github.io/polochau/" target="_blank">Polo Chau</a>
						am Georgia Institute of Technology entwickelt.

						Übersetzt und angepasst vom <a href="https://bil-hamburg.de" target="_blank">Business Innovation Lab</a> 
						der Hochschule für Angewandte Wissenschaften Hamburg im Rahmen des <a href="https://digitalzentrum-hamburg.de/" target="_blank">
						Mittelstand Digital-Zentrum Hamburg</a>.</p>
				</p>
			</div>
					</p>
				</div>
			</div>
		</div>
	</div>
</div>

<style lang="scss">
	a {
		color: theme('colors.blue.500');

		&:hover {
			color: theme('colors.blue.700');
		}
	}

	.bold-purple {
		color: theme('colors.purple.700');
		font-weight: bold;
	}

	code {
		color: theme('colors.gray.500');
		background-color: theme('colors.gray.50');
		font-family: theme('fontFamily.mono');
	}

	.q-color {
		color: theme('colors.blue.400');
	}

	.k-color {
		color: theme('colors.red.400');
	}

	.v-color {
		color: theme('colors.green.400');
	}

	.purple-color {
		color: theme('colors.purple.500');
	}

	.article-section {
		padding-bottom: 2rem;
	}
	.architecture-section {
		padding-top: 1rem;
	}
	.video-container {
		position: relative;
		padding-bottom: 56.25%; /* 16:9 aspect ratio */
		height: 0;
		overflow: hidden;
		max-width: 100%;
		background: #000;
	}

	.video-container iframe {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
	}

	#description {
		padding-bottom: 3rem;
		margin-left: auto;
		margin-right: auto;
		max-width: 78ch;
	}

	#description h1 {
		color: theme('colors.purple.700');
		font-size: 2.2rem;
		font-weight: 300;
		padding-top: 1rem;
	}

	#description h2 {
		// color: #444;
		color: theme('colors.purple.700');
		font-size: 2rem;
		font-weight: 300;
		padding-top: 1rem;
	}

	#description h3 {
		color: theme('colors.gray.700');
		font-size: 1.6rem;
		font-weight: 200;
		padding-top: 1rem;
	}

	#description h4 {
		color: theme('colors.gray.700');
		font-size: 1.6rem;
		font-weight: 200;
		padding-top: 1rem;
	}

	#description p {
		margin: 1rem 0;
	}

	#description p img {
		vertical-align: middle;
	}

	#description .figure-caption {
		font-size: 0.8rem;
		margin-top: 0.5rem;
		text-align: center;
		margin-bottom: 2rem;
	}

	#description ol {
		margin-left: 3rem;
		list-style-type: decimal;
	}

	#description li {
		margin: 0.6rem 0;
	}

	#description p,
	#description div,
	#description li {
		color: theme('colors.gray.600');
		// font-size: 17px;
		// font-size: 15px;
		line-height: 1.6;
	}

	#description small {
		font-size: 0.8rem;
	}

	#description ol li img {
		vertical-align: middle;
	}

	#description .video-link {
		color: theme('colors.blue.600');
		cursor: pointer;
		font-weight: normal;
		text-decoration: none;
	}

	#description ul {
		list-style-type: disc;
		margin-left: 2.5rem;
		margin-bottom: 1rem;
	}

	#description a:hover,
	#description .video-link:hover {
		text-decoration: underline;
	}

	.figure,
	.video {
		width: 100%;
		display: flex;
		flex-direction: column;
		align-items: center;
	}
</style>
