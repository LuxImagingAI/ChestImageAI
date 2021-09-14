from collections import defaultdict

import sys
sys.path.append('/home/users/jsoelter/Code/segmentation_models/segmentation_models.pytorch/')

from contextlib import nullcontext
import torch
import segmentation_models_pytorch as smp

import train_utils
#from segmentation_models_pytorch.decoder import UnetDecoder
#from segmentation_models_pytorch.base import SegmentationHead, initialization


class FeatureExtractor(torch.nn.Module):
    '''Returns features from different layers of a network'''
    
    def __init__(self, backbone, link_layers = None, out_channels = None, include_image = False):
        super().__init__()
        
        self.backbone = backbone
        self.name = getattr(backbone, 'name', '') + 'Feature'
        
        self.layers = dict([*self.backbone.named_modules()])
            
        # extract model info from timm if available
        if hasattr(backbone, 'feature_info'):
            self.info_dict = {d['module']: d for d in backbone.feature_info}

        self._features = {}
        self.out_channels = {}

        if link_layers == 'timm':
            link_layers = self.info_dict.keys()
            out_channels = [self.info_dict[k]['num_chs'] for k in link_layers]       
        if link_layers is not None:
            self.register_link_layer(link_layers, out_channels)
        
        self.include_image = include_image
        if include_image: #
            self._features['image'] = None
            self.out_channels['image'] = None


    def save_outputs_hook(self, layer_name):
        def fn(_, __, output):
            self._features[layer_name] = output
        return fn

    
    def register_link_layer(self, link_layers, out_channels):
        for layer_name, out_channel in zip(link_layers, out_channels):
            # add layer to features if layer is not already included
            if layer_name not in self._features:
                layer = self.layers[layer_name]
                layer.register_forward_hook(self.save_outputs_hook(layer_name))
                self._features[layer_name] = None
                self.out_channels[layer_name] = out_channel
        
    
    def get_features(self):
        return self._features
    

    def forward(self, data):
        if self.include_image:
            self._features['image'] = data.get('image', None)
        _ = self.backbone(data)
        return self.get_features()


class UnetDecoder(torch.nn.Module):
    """   """

    def __init__(
        self,
        # encoder parameters
        encoder_channels,
        # decoder parameters
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32, 16),
        decoder_attention_type = None,
        # classification parameters
        classes = 1,
        activation = None,
        name = None
        ):
        
        super().__init__()

        self.decoder = smp.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(encoder_channels)-1,
            use_batchnorm=decoder_use_batchnorm,
            center = False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.name = "uNet{}".format(name)
        self.initialize()
    
    
    def initialize(self):
        
        smp.base.initialization.initialize_decoder(self.decoder)
        smp.base.initialization.initialize_head(self.segmentation_head)

    
    def forward(self, feature_list):
        """Sequentially pass `featurelist` trough decoder and heads"""

        decoder_output = self.decoder(*feature_list)
        masks = self.segmentation_head(decoder_output)

        return masks


class RegionalAttentionPooling(torch.nn.Module):
    """ Regional attention Layer"""
    
    def __init__(self, in_channel, n_heads):
        super(RegionalAttentionPooling, self).__init__()
        self.channel_in = in_channel
        self.n_heads = n_heads
        
        self.attention = torch.nn.Conv2d(in_channels = in_channel , out_channels = n_heads , kernel_size= 1)
        self.softmax  = torch.nn.Softmax(dim=-1)
        
        self._reset_parameters()

        
    def forward(self, x):
        """
            inputs :
                x : list of single input feature map [(B X C X W X H)]
            returns :
                out : self attention value + input feature 
                attention: B X H X C (H heads)
        """
        assert len(x) == 1, 'only single feature layers are allowed'
        x =  x[0]
        n_batch, n_channel, width, height = x.size()
        n_spatial = width*height
        
        attention_logits = self.attention(x)
        attention = self.softmax(attention_logits.view(n_batch, self.n_heads, n_spatial))
        
        # b:batch, h:n_heads, s:spatial(H*W), channel
        out = torch.einsum('bhs,bcs->bhc', attention, x.view(n_batch, n_channel, n_spatial))
        
        return out #, attention
    
    
    def _reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.attention.weight)
        self.attention.bias.data.fill_(0.01)


class BrixiaScore(torch.nn.Module):
    
    def __init__(self, encoder_channels, num_classes=2, n_heads=1):
        
        super().__init__()
        
        assert len(encoder_channels) == 1, 'Can only connect on laye to head' 
        encoder_channels = encoder_channels[0]
        self.pool = RegionalAttentionPooling(in_channel=encoder_channels, n_heads=n_heads)
        self.cls = torch.nn.Linear(encoder_channels, num_classes)

    
    def forward(self, features):
        
        regional_feature = self.pool(features)
        score = self.cls(regional_feature)
        #convert to batch, class, head
        score = score.permute(0,2,1)

        return score


class ClassificationHead(torch.nn.Module):
    
    def __init__(self, encoder_channels, num_classes=1):
        
        super().__init__()
        assert len(encoder_channels) == 1, 'Currently only one feature vector is supported'

        self.classifier = torch.nn.Conv2d(encoder_channels[0], num_classes, kernel_size=(1, 1), stride=(1, 1))
        torch.nn.init.zeros_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)
        
    def forward(self, feature_vector):
            
        pred = self.classifier(feature_vector[0])
        assert pred.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return pred[...,0,0]
    
    
class AttachedHead(torch.nn.Module):
    """ create head from config and attache to feature_extractor"""
    
    def __init__(self, feature_extractor, config):
        '''
            parameter:
                feature_extractor = FeatureExtraktor to link the head
                config = head configuration
        
        '''
        
        super().__init__()
        feature_extractor.register_link_layer(**config['features'])
        self.feature_extractor = feature_extractor
        self.link_layers = config['features']['link_layers']
        self.head = train_utils.instantiate_object(
            encoder_channels = [self.feature_extractor.out_channels[k] for k in self.link_layers],
            **config['model'],
        )

    def forward(self, data=None, meta = None, backbone_grad=False):
        ''' executes head, if `data` is given executes backbone first. 
        
        `meta`: tensor with meta feature to injected before the head
        `backbone_grad`: if autograd is calculated on backbone (default: False)
        '''
        
        if data is not None: # run the backbone and get features
            # switch gradient calculation on/off
            gradient_context = nullcontext if backbone_grad else torch.no_grad 
            with gradient_context():
                features = self.feature_extractor(data)
        else: # extract the features of last backone run
            features = self.feature_extractor.get_features()
        relevant_features = [features[k] for k in self.link_layers]
        if meta is not None: # add meta to features
            relevant_features = [torch.cat((f, meta), 1) for f in relevant_features]
        out = self.head(relevant_features)
        return out
    

class MultiHeadEvaluator():
    
    def __init__(self, losses, device=None):
    
        self.losses = losses
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    def __call__(self, backbone, heads, loaders, num_iter=-1):
        
        with torch.no_grad():
            loss_values = defaultdict(list)
            for k, v in heads.items():
                #f = v['features']
                m = v['model']
                m.eval()
                backbone.eval()
                for i, batch in enumerate(loaders[k]):
                    x = batch[v['input']].float().to(self.device)
                    y = batch[v['target']].long().to(self.device)
                    _ = backbone(x)
                    pred = m()
                    loss = self.losses[k](pred, y)
                    loss_values[k].append(loss.cpu().numpy())
                    if i == num_iter-1:
                        break
                
        return loss_values
    

    def predictions(backbone, head, loaders, device):

        with torch.no_grad():

            #f = head['features']
            m = head['model'].head
            m.eval()
            backbone.eval()
            preds, targets = [], []
            for i, batch in enumerate(loaders[key]):
                x = batch[head['input']].float().to(device)
                y = batch[head['target']]
                _ = backbone(x)
                pred = m()
                preds.append(pred.to('cpu').numpy())
                targets.append(y)

        return np.vstack(preds), np.vstack(targets)