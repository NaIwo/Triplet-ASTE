from typing import Dict, List
from typing import Optional

from .base_model import BaseModel
from .base_triplet_model import BaseTripletModel
from .model_elements.span_aggregators import (
    BaseAggregator,
    EndPointAggregator,
    MaxPoolAggregator
)
from .model_elements.embeddings import BaseEmbedding, TransformerWithAggregation
from .outputs import (
    BaseModelOutput,
    SpanCreatorOutput,
    TripletModelOutput,
    SentimentModelOutput,
    ModelOutput,
    ClassificationModelOutput,
    FinalTriplets,
    FinalMetric
)
from .specialty_models import (
    EmbeddingsExtenderModel,
    MetricTripletExtractorModel,
    SpanClassifierModel,
    NeuralTripletExtractorModel,
    NonSentimentNeuralTripletExtractorModel,
    NonSentimentAttentionTripletExtractorModel,
    PairClassifierModel
)
from .specialty_models import (
    SpanCreatorModel,
    NonSentimentMetricTripletExtractorModel,
    SentimentPredictor
)
from ..dataset.reader import Batch


class OpinionBasedTripletModel(BaseTripletModel):
    def __init__(self, model_name='Opinion Based Triplet Model', config: Optional[Dict] = None, *args, **kwargs):
        super(OpinionBasedTripletModel, self).__init__(model_name, config=config)

        self.span_creator: BaseModel = SpanCreatorModel(input_dim=self.emb_layer.embedding_dim, config=config)
        self.aggregator: BaseAggregator = EndPointAggregator(input_dim=self.emb_layer.embedding_dim, config=config)
        self.sentiment_extender: BaseModel = EmbeddingsExtenderModel(input_dim=self.aggregator.output_dim,
                                                                     config=config)
        self.triplets_extractor: BaseModel = MetricTripletExtractorModel(
            config=config, input_dim=self.aggregator.output_dim
        )

        self.final_metrics = FinalMetric()

        self.model_with_losses = {
            self.span_creator: 'span_creator_output',
            self.triplets_extractor: 'triplet_output'
        }
        self.model_with_metrics = {
            self.span_creator: 'span_creator_output',
            self.triplets_extractor: 'triplet_output',
            self.final_metrics: 'final_triplet'
        }

    def forward(self, batch: Batch) -> ModelOutput:
        batch.to_device(self.device)
        emb_output: BaseModelOutput = self.emb_layer(batch)
        span_creator_output: SpanCreatorOutput = self.span_creator(emb_output)

        span_creator_output.aspects_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_aspect_span_predictions()
        )
        span_creator_output.opinions_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_opinion_span_predictions()
        )
        extended_opinions: SentimentModelOutput = self.sentiment_extender(span_creator_output.opinions_agg_emb)
        span_creator_output = span_creator_output.extend_opinions_with_sentiments(extended_opinions)

        triplet_output: TripletModelOutput = self.triplets_extractor(span_creator_output)

        final_triplet = FinalTriplets(
            batch=batch,
            pred_triplets=triplet_output.get_triplets(),
        )

        return ModelOutput(
            batch=batch,
            span_creator_output=span_creator_output,
            triplet_output=triplet_output,
            final_triplet=final_triplet
        )

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.emb_layer.parameters(), 'lr': self.config['model']['transformer']['lr']},
            {'params': self.span_creator.parameters(), 'lr': self.config['model']['span-creator']['lr']},
            {'params': self.aggregator.get_parameters(), 'lr': self.config['model']['aggregator']['lr']},
            {'params': self.sentiment_extender.parameters(), 'lr': self.config['model']['sentiment-extender']['lr']},
            {'params': self.triplets_extractor.parameters(), 'lr': self.config['model']['triplet-extractor']['lr']},
        ]


class OpinionBasedTripletTwoEmbeddersModel(OpinionBasedTripletModel):
    def __init__(self, model_name='Opinion Based Triplet Model with Two Embedders',
                 config: Optional[Dict] = None, *args, **kwargs):
        super(OpinionBasedTripletTwoEmbeddersModel, self).__init__(model_name, config=config)
        self.matrix_emb_layer: BaseEmbedding = TransformerWithAggregation(config=config)

    def forward(self, batch: Batch) -> ModelOutput:
        batch.to_device(self.device)
        emb_output: BaseModelOutput = self.emb_layer(batch)
        matrix_emb_output: BaseModelOutput = self.matrix_emb_layer(batch)

        span_creator_output: SpanCreatorOutput = self.span_creator(emb_output)

        span_creator_output.aspects_agg_emb = self.aggregator.aggregate(
            matrix_emb_output.features,
            span_creator_output.get_aspect_span_predictions()
        )
        span_creator_output.opinions_agg_emb = self.aggregator.aggregate(
            matrix_emb_output.features,
            span_creator_output.get_opinion_span_predictions()
        )
        extended_opinions: SentimentModelOutput = self.sentiment_extender(span_creator_output.opinions_agg_emb)
        span_creator_output = span_creator_output.extend_opinions_with_sentiments(extended_opinions)

        triplet_output: TripletModelOutput = self.triplets_extractor(span_creator_output)

        final_triplet = FinalTriplets(
            batch=batch,
            pred_triplets=triplet_output.get_triplets(),
        )

        return ModelOutput(
            batch=batch,
            span_creator_output=span_creator_output,
            triplet_output=triplet_output,
            final_triplet=final_triplet
        )

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.emb_layer.parameters(), 'lr': self.config['model']['transformer']['lr']},
            {'params': self.matrix_emb_layer.parameters(), 'lr': self.config['model']['transformer']['lr']},
            {'params': self.span_creator.parameters(), 'lr': self.config['model']['span-creator']['lr']},
            {'params': self.aggregator.get_parameters(), 'lr': self.config['model']['aggregator']['lr']},
            {'params': self.sentiment_extender.parameters(), 'lr': self.config['model']['sentiment-extender']['lr']},
            {'params': self.triplets_extractor.parameters(), 'lr': self.config['model']['triplet-extractor']['lr']},
        ]


class OpinionBasedTripletModelClassifier(BaseTripletModel):
    def __init__(self,
                 model_name='Opinion Based Triplet Model with Classifier',
                 config: Optional[Dict] = None, *args, **kwargs):
        super(OpinionBasedTripletModelClassifier, self).__init__(model_name, config=config)

        self.span_creator: BaseModel = SpanCreatorModel(input_dim=self.emb_layer.embedding_dim, config=config)
        self.aggregator: BaseAggregator = MaxPoolAggregator(input_dim=self.emb_layer.embedding_dim, config=config)
        self.sentiment_extender: BaseModel = EmbeddingsExtenderModel(input_dim=self.aggregator.output_dim,
                                                                     config=config)
        self.span_classifier: BaseModel = SpanClassifierModel(input_dim=self.aggregator.output_dim, config=config)
        self.pair_classifier: BaseModel = PairClassifierModel(input_dim=self.sentiment_extender.output_dim, config=config)
        self.triplets_extractor: BaseModel = MetricTripletExtractorModel(
            config=config, input_dim=self.sentiment_extender.output_dim
        )

        self.final_metrics = FinalMetric()

        self.model_with_losses = {
            self.span_creator: 'span_creator_output',
            self.triplets_extractor: 'triplet_output',
            self.span_classifier: 'span_classifier_output',
            self.pair_classifier: 'pair_output'
        }
        self.model_with_metrics = {
            self.span_creator: 'span_creator_output',
            self.triplets_extractor: 'triplet_output',
            self.pair_classifier: 'pair_output',
            self.span_classifier: 'span_classifier_output',
            self.final_metrics: 'final_triplet'
        }

    def forward(self, batch: Batch) -> ModelOutput:
        batch.to_device(self.device)
        emb_output: BaseModelOutput = self.emb_layer(batch)
        span_creator_output: SpanCreatorOutput = self.span_creator(emb_output)

        span_creator_output.aspects_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_aspect_span_predictions()
        )
        span_creator_output.opinions_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_opinion_span_predictions()
        )
        span_creator_output.sentence_emb = emb_output.features.clone()

        span_classifier_output: ClassificationModelOutput = self.span_classifier(span_creator_output)

        extended_opinions: SentimentModelOutput = self.sentiment_extender(span_creator_output.opinions_agg_emb)
        span_creator_output = span_creator_output.extend_opinions_with_sentiments(extended_opinions)

        triplet_output: TripletModelOutput = self.triplets_extractor(span_creator_output)

        pair_output = self.pair_classifier(triplet_output)

        final_triplet = FinalTriplets(
            batch=batch,
            pred_triplets=pair_output.get_triplets(),
        )

        return ModelOutput(
            batch=batch,
            final_triplet=final_triplet,
            span_creator_output=span_creator_output,
            triplet_output=triplet_output,
            pair_output=pair_output,
            span_classifier_output=span_classifier_output
        )

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.emb_layer.parameters(), 'lr': self.config['model']['transformer']['lr']},
            {'params': self.span_creator.parameters(), 'lr': self.config['model']['span-creator']['lr']},
            {'params': self.aggregator.get_parameters(), 'lr': self.config['model']['aggregator']['lr']},
            {'params': self.span_classifier.parameters(), 'lr': self.config['model']['span-classifier']['lr']},
            {'params': self.pair_classifier.parameters(), 'lr': self.config['model']['pair-classifier']['lr']},
            {'params': self.sentiment_extender.parameters(), 'lr': self.config['model']['sentiment-extender']['lr']},
            {'params': self.triplets_extractor.parameters(), 'lr': self.config['model']['triplet-extractor']['lr']},
        ]


class SentimentPredictorTripletModel(BaseTripletModel):
    def __init__(self, model_name='Sentiment Predictor Triplet Model', config: Optional[Dict] = None, *args, **kwargs):
        super(SentimentPredictorTripletModel, self).__init__(model_name, config=config)

        self.span_creator: BaseModel = SpanCreatorModel(
            input_dim=self.emb_layer.embedding_dim, config=config
        )
        self.aggregator: BaseAggregator = MaxPoolAggregator(input_dim=self.emb_layer.embedding_dim, config=config)
        self.triplets_extractor: BaseModel = NonSentimentMetricTripletExtractorModel(
            config=config, input_dim=self.aggregator.output_dim
        )
        self.span_classifier: BaseModel = SpanClassifierModel(input_dim=self.aggregator.output_dim, config=config)
        self.sentiment_predictor: BaseModel = SentimentPredictor(
            input_dim=self.aggregator.output_dim,
            config=config
        )

        self.final_metrics = FinalMetric()

        self.model_with_losses = {
            self.span_creator: 'span_creator_output',
            self.span_classifier: 'span_classifier_output',
            self.triplets_extractor: 'triplet_output',
            self.sentiment_predictor: 'predictor_triplet_output'
        }
        self.model_with_metrics = {
            self.span_creator: 'span_creator_output',
            self.span_classifier: 'span_classifier_output',
            self.triplets_extractor: 'triplet_output',
            self.sentiment_predictor: 'predictor_triplet_output',
            self.final_metrics: 'final_triplet'
        }

    def forward(self, batch: Batch) -> ModelOutput:
        batch.to_device(self.device)
        emb_output: BaseModelOutput = self.emb_layer(batch)
        span_creator_output: SpanCreatorOutput = self.span_creator(emb_output)

        span_creator_output.aspects_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_aspect_span_predictions()
        )
        span_creator_output.opinions_agg_emb = self.aggregator.aggregate(
            emb_output.features,
            span_creator_output.get_opinion_span_predictions()
        )
        span_creator_output.sentence_emb = emb_output.features.clone()

        span_classifier_output: ClassificationModelOutput = self.span_classifier(span_creator_output)

        triplet_output: TripletModelOutput = self.triplets_extractor(span_creator_output)

        predictor_triplet_output: TripletModelOutput = self.sentiment_predictor(triplet_output)

        final_triplet = FinalTriplets(
            batch=batch,
            pred_triplets=predictor_triplet_output.get_triplets(),
        )

        return ModelOutput(
            batch=batch,
            span_creator_output=span_creator_output,
            span_classifier_output=span_classifier_output,
            triplet_output=triplet_output,
            predictor_triplet_output=predictor_triplet_output,
            final_triplet=final_triplet
        )

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.emb_layer.parameters(), 'lr': self.config['model']['transformer']['lr']},
            {'params': self.span_creator.parameters(), 'lr': self.config['model']['span-creator']['lr']},
            {'params': self.span_classifier.parameters(), 'lr': self.config['model']['span-classifier']['lr']},
            {'params': self.aggregator.get_parameters(), 'lr': self.config['model']['aggregator']['lr']},
            {'params': self.sentiment_predictor.parameters(), 'lr': self.config['model']['sentiment-predictor']['lr']},
            {'params': self.triplets_extractor.parameters(), 'lr': self.config['model']['triplet-extractor']['lr']},
        ]


class SentimentPredictorTripletTwoEmbeddersModel(SentimentPredictorTripletModel):
    def __init__(self, model_name='Sentiment Predictor Triplet Model with Two Embedders',
                 config: Optional[Dict] = None, *args, **kwargs):
        super(SentimentPredictorTripletTwoEmbeddersModel, self).__init__(model_name, config=config)

        self.matrix_emb_layer: BaseEmbedding = TransformerWithAggregation(config=config)

    def forward(self, batch: Batch) -> ModelOutput:
        batch.to_device(self.device)
        emb_output: BaseModelOutput = self.emb_layer(batch)
        matrix_emb_output: BaseModelOutput = self.matrix_emb_layer(batch)

        span_creator_output: SpanCreatorOutput = self.span_creator(emb_output)

        span_creator_output.aspects_agg_emb = self.aggregator.aggregate(
            matrix_emb_output.features,
            span_creator_output.get_aspect_span_predictions()
        )
        span_creator_output.opinions_agg_emb = self.aggregator.aggregate(
            matrix_emb_output.features,
            span_creator_output.get_opinion_span_predictions()
        )

        span_classifier_output: ClassificationModelOutput = self.span_classifier(span_creator_output)

        triplet_output: TripletModelOutput = self.triplets_extractor(span_creator_output)

        predictor_triplet_output: TripletModelOutput = self.sentiment_predictor(triplet_output)

        final_triplet = FinalTriplets(
            batch=batch,
            pred_triplets=predictor_triplet_output.get_triplets(),
        )

        return ModelOutput(
            batch=batch,
            span_creator_output=span_creator_output,
            triplet_output=triplet_output,
            predictor_triplet_output=predictor_triplet_output,
            span_classifier_output=span_classifier_output,
            final_triplet=final_triplet
        )

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.emb_layer.parameters(), 'lr': self.config['model']['transformer']['lr']},
            {'params': self.matrix_emb_layer.parameters(), 'lr': self.config['model']['transformer']['lr']},
            {'params': self.span_creator.parameters(), 'lr': self.config['model']['span-creator']['lr']},
            {'params': self.aggregator.get_parameters(), 'lr': self.config['model']['aggregator']['lr']},
            {'params': self.sentiment_predictor.parameters(), 'lr': self.config['model']['sentiment-predictor']['lr']},
            {'params': self.triplets_extractor.parameters(), 'lr': self.config['model']['triplet-extractor']['lr']},
            {'params': self.span_classifier.parameters(), 'lr': self.config['model']['span-classifier']['lr']},
        ]