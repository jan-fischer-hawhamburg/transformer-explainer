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
						zielt der MLP darauf ab, die Darstellung jedes Tokens zu verfeinern.

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
				Sie verwandelt den Text in eine numerische Darstellung, mit der das Modell arbeiten kann. 
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
				die zeigt, wie die Eingabeaufforderung in eine Vektordarstellung umgewandelt wird. Der Prozess umfasst 
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
					können wir ihre Vektordarstellung aus den Einbettungen erhalten.
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
              	Jede Position in einer Sequenz wird durch eine einzigartige mathematische Darstellung 
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
					Abschließend summieren wir die Token- und Positionskodierungen, um die endgültige Einbettungsdarstellung zu erhalten. 
					Diese kombinierte Darstellung erfasst sowohl die semantische Bedeutung 
					der Tokens als auch ihre Position in der Eingabesequenz.
				</p>
			</div>
		</div>

		<div class="architecture-section">
			<h2>Transformer Block</h2>

			<p>	Der Kern der Verarbeitung im Transformer liegt im Transformer-Block, 
				der aus einer Multi-Head-Selbstaufmerksamkeit (multi-head self-attention) und einer Multi-Layer-Perceptron-Schicht besteht. 
				Die meisten Modelle bestehen aus mehreren solcher Blöcke, die nacheinander sequenziell gestapelt sind. 
				Die Token-Darstellungen entwickeln sich durch die Schichten, vom ersten Block bis zum zwölften, 
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
					<h4>Step 1: Query, Key, und Value Matrizen</h4>

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
					<h4>Step 2: Masked Self-Attention</h4>
					<p>
						Masked self-attention allows the model to generate sequences by focusing on relevant
						parts of the input while preventing access to future tokens.
					</p>

					<div class="figure">
						<img src="./article_assets/attention.png" width="80%" align="middle" />
					</div>
					<div class="figure-caption">
						Figure <span class="attention">3</span>. Using Query, Key, and Value matrices to
						calculate masked self-attention.
					</div>

					<ul>
						<li>
							<strong>Attention Score</strong>: The dot product of
							<span class="q-color">Query</span>
							and <span class="k-color">Key</span> matrices determines the alignment of each query with
							each key, producing a square matrix that reflects the relationship between all input tokens.
						</li>
						<li>
							<strong>Masking</strong>: A mask is applied to the upper triangle of the attention
							matrix to prevent the model from accessing future tokens, setting these values to
							negative infinity. The model needs to learn how to predict the next token without
							“peeking” into the future.
						</li>
						<li>
							<strong>Softmax</strong>: After masking, the attention score is converted into
							probability by the softmax operation which takes the exponent of each attention score.
							Each row of the matrix sums up to one and indicates the relevance of every other token
							to the left of it.
						</li>
					</ul>
				</div>
				<div class="article-subsection-l2">
					<h4>Step 3: Output</h4>
					<p>
						The model uses the masked self-attention scores and multiplies them with the
						<span class="v-color">Value</span> matrix to get the
						<span class="purple-color">final output</span>
						of the self-attention mechanism. GPT-2 has <code>12</code> self-attention heads, each capturing
						different relationships between tokens. The outputs of these heads are concatenated and passed
						through a linear projection.
					</p>
				</div>

				<div class="article-subsection" id="article-activation">
					<h3>MLP: Multi-Layer Perceptron</h3>

					<div class="figure">
						<img src="./article_assets/mlp.png" width="70%" align="middle" />
					</div>
					<div class="figure-caption">
						Figure <span class="attention">4</span>. Using MLP layer to project the self-attention
						representations into higher dimensions to enhance the model's representational capacity.
					</div>

					<p>
						After the multiple heads of self-attention capture the diverse relationships between the
						input tokens, the concatenated outputs are passed through the Multilayer Perceptron
						(MLP) layer to enhance the model's representational capacity. The MLP block consists of
						two linear transformations with a GELU activation function in between. The first linear
						transformation increases the dimensionality of the input four-fold from <code>768</code>
						to <code>3072</code>. The second linear transformation reduces the dimensionality back
						to the original size of <code>768</code>, ensuring that the subsequent layers receive
						inputs of consistent dimensions. Unlike the self-attention mechanism, the MLP processes
						tokens independently and simply map them from one representation to another.
					</p>
				</div>

				<div class="architecture-section" id="article-prob">
					<h2>Output Probabilities</h2>
					<p>
						After the input has been processed through all Transformer blocks, the output is passed
						through the final linear layer to prepare it for token prediction. This layer projects
						the final representations into a <code>50,257</code>
						dimensional space, where every token in the vocabulary has a corresponding value called
						<code>logit</code>. Any token can be the next word, so this process allows us to simply
						rank these tokens by their likelihood of being that next word. We then apply the softmax
						function to convert the logits into a probability distribution that sums to one. This
						will allow us to sample the next token based on its likelihood.
					</p>

					<div class="figure">
						<img src="./article_assets/softmax.png" width="60%" align="middle" />
					</div>
					<div class="figure-caption">
						Figure <span class="attention">5</span>. Each token in the vocabulary is assigned a
						probability based on the model's output logits. These probabilities determine the
						likelihood of each token being the next word in the sequence.
					</div>

					<p id="article-temperature">
						The final step is to generate the next token by sampling from this distribution The <code
							>temperature</code
						>
						hyperparameter plays a critical role in this process. Mathematically speaking, it is a very
						simple operation: model output logits are simply divided by the
						<code>temperature</code>:
					</p>

					<ul>
						<li>
							<code>temperature = 1</code>: Dividing logits by one has no effect on the softmax
							outputs.
						</li>
						<li>
							<code>temperature &lt; 1</code>: Lower temperature makes the model more confident and
							deterministic by sharpening the probability distribution, leading to more predictable
							outputs.
						</li>
						<li>
							<code>temperature &gt; 1</code>: Higher temperature creates a softer probability
							distribution, allowing for more randomness in the generated text – what some refer to
							as model <em>“creativity”</em>.
						</li>
					</ul>

					<p>
						Adjust the temperature and see how you can balance between deterministic and diverse
						outputs!
					</p>
				</div>

				<div class="architecture-section">
					<h2>Advanced Architectural Features</h2>

					<p>
						There are several advanced architectural features that enhance the performance of
						Transformer models. While important for the model's overall performance, they are not as
						important for understanding the core concepts of the architecture. Layer Normalization,
						Dropout, and Residual Connections are crucial components in Transformer models,
						particularly during the training phase. Layer Normalization stabilizes training and
						helps the model converge faster. Dropout prevents overfitting by randomly deactivating
						neurons. Residual Connections allows gradients to flow directly through the network and
						helps to prevent the vanishing gradient problem.
					</p>
					<div class="article-subsection" id="article-ln">
						<h3>Layer Normalization</h3>

						<p>
							Layer Normalization helps to stabilize the training process and improves convergence.
							It works by normalizing the inputs across the features, ensuring that the mean and
							variance of the activations are consistent. This normalization helps mitigate issues
							related to internal covariate shift, allowing the model to learn more effectively and
							reducing the sensitivity to the initial weights. Layer Normalization is applied twice
							in each Transformer block, once before the self-attention mechanism and once before
							the MLP layer.
						</p>
					</div>
					<div class="article-subsection" id="article-dropout">
						<h3>Dropout</h3>

						<p>
							Dropout is a regularization technique used to prevent overfitting in neural networks
							by randomly setting a fraction of model weights to zero during training. This
							encourages the model to learn more robust features and reduces dependency on specific
							neurons, helping the network generalize better to new, unseen data. During model
							inference, dropout is deactivated. This essentially means that we are using an
							ensemble of the trained subnetworks, which leads to a better model performance.
						</p>
					</div>
					<div class="article-subsection" id="article-residual">
						<h3>Residual Connections</h3>

						<p>
							Residual connections were first introduced in the ResNet model in 2015. This
							architectural innovation revolutionized deep learning by enabling the training of very
							deep neural networks. Essentially, residual connections are shortcuts that bypass one
							or more layers, adding the input of a layer to its output. This helps mitigate the
							vanishing gradient problem, making it easier to train deep networks with multiple
							Transformer blocks stacked on top of each other. In GPT-2, residual connections are
							used twice within each Transformer block: once before the MLP and once after, ensuring
							that gradients flow more easily, and earlier layers receive sufficient updates during
							backpropagation.
						</p>
					</div>
				</div>

				<div class="article-section">
					<h1>Interactive Features</h1>
					<p>
						Transformer Explainer is built to be interactive and allows you to explore the inner
						workings of the Transformer. Here are some of the interactive features you can play
						with:
					</p>

					<ul>
						<li>
							<strong>Input your own text sequence</strong> to see how the model processes it and predicts
							the next word. Explore attention weights, intermediate computations, and see how the final
							output probabilities are calculated.
						</li>
						<li>
							<strong>Use the temperature slider</strong> to control the randomness of the model’s predictions.
							Explore how you can make the model output more deterministic or more creative by changing
							the temperature value.
						</li>
						<li>
							<strong>Interact with attention maps</strong> to see how the model focuses on different
							tokens in the input sequence. Hover over tokens to highlight their attention weights and
							explore how the model captures context and relationships between words.
						</li>
					</ul>
				</div>

				<div class="article-section">
					<h2>Video Tutorial</h2>
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
					<h2>How is Transformer Explainer Implemented?</h2>
					<p>
						Transformer Explainer features a live GPT-2 (small) model running directly in the
						browser. This model is derived from the PyTorch implementation of GPT by Andrej
						Karpathy's
						<a href="https://github.com/karpathy/nanoGPT" title="Github" target="_blank"
							>nanoGPT project</a
						>
						and has been converted to
						<a href="https://onnxruntime.ai/" title="ONNX" target="_blank">ONNX Runtime</a>
						for seamless in-browser execution. The interface is built using JavaScript, with
						<a href="https://kit.svelte.dev/" title="Svelte" target="_blank">Svelte</a>
						as a front-end framework and
						<a href="http://D3.js" title="D3" target="_blank">D3.js</a>
						for creating dynamic visualizations. Numerical values are updated live following the user
						input.
					</p>
				</div>

				<div class="article-section">
					<h2>Who developed the Transformer Explainer?</h2>
					<p>
						Transformer Explainer was created by

						<a href="https://aereeeee.github.io/" target="_blank">Aeree Cho</a>,
						<a href="https://www.linkedin.com/in/chaeyeonggracekim/" target="_blank">Grace C. Kim</a
						>,
						<a href="https://alexkarpekov.com/" target="_blank">Alexander Karpekov</a>,
						<a href="https://alechelbling.com/" target="_blank">Alec Helbling</a>,
						<a href="https://zijie.wang/" target="_blank">Jay Wang</a>,
						<a href="https://seongmin.xyz/" target="_blank">Seongmin Lee</a>,
						<a href="https://bhoov.com/" target="_blank">Benjamin Hoover</a>, and
						<a href="https://poloclub.github.io/polochau/" target="_blank">Polo Chau</a>

						at the Georgia Institute of Technology.
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
